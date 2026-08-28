// lib/memory-summarizer.ts
// Auto-summarization engine: summarizes short-term events into long-term memories.
// Trigger: every N events (configurable). Short-term events are NOT deleted after summarization.

import type { MemoryEntry } from "./memory-types";
import { DEFAULT_SUMMARIZATION_PROMPT } from "./memory-types";
import {
    loadMemoryConfig,
    loadMemoryEntries,
    saveMemoryEntry,
    deleteMemoryEntries,
    getEventCounter,
    resetEventCounter,
    getLastSummarizedTimestamp,
    setLastSummarizedTimestamp,
    incrementCoreMemoryCounter,
} from "./memory-storage";
import { resolveMemoryEmbeddingApiConfig, resolveMemorySummaryApiConfig } from "./memory-api-config";
import { loadNativeTimeline, formatTimelineForSummarization, filterTimelineByAllowedSources } from "./short-term-assembler";
import { generateEmbedding, generateEmbeddings, cosineSimilarity, keywordOverlapRatio } from "./memory-embedding";
import { simpleLLMCall } from "./api-helpers";
import { maybeRunCoreMemoryPipeline } from "./core-memory-builder";
import { jsonrepair } from "jsonrepair";

/** Per-character lock to prevent concurrent summarization. */
const summarizingSet = new Set<string>();

type ExtractedFact = {
    content: string;
    importance: number;
    tags: string[];
    entities: string[];
};

type MemoryExtraction = {
    summary: string;
    facts: ExtractedFact[];
};

const STRUCTURED_OUTPUT_SUFFIX = `

输出必须是一个完整 JSON 对象，不要使用 Markdown 代码块：
{"summary":"本批事件的简洁总结","facts":[{"content":"可独立成立的单一事实","importance":0.8,"tags":["关键词"],"entities":["人物或事物"]}]}
facts 最多 8 条；没有值得长期保留的事实时返回空数组。`;

function stringList(value: unknown, max: number): string[] {
    if (!Array.isArray(value)) return [];
    return Array.from(new Set(value.map(String).map(item => item.trim()).filter(Boolean))).slice(0, max);
}

function parseMemoryExtraction(raw: string): MemoryExtraction {
    const cleaned = raw.trim().replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/, "");
    try {
        const parsed = JSON.parse(jsonrepair(cleaned)) as Record<string, unknown>;
        const summary = String(parsed.summary ?? "").trim();
        const facts = (Array.isArray(parsed.facts) ? parsed.facts : [])
            .map(item => {
                const fact = (item && typeof item === "object" ? item : {}) as Record<string, unknown>;
                const content = String(fact.content ?? "").trim();
                const rawImportance = Number(fact.importance);
                return {
                    content,
                    importance: Number.isFinite(rawImportance) ? Math.max(0.1, Math.min(1, rawImportance)) : 0.75,
                    tags: stringList(fact.tags, 5),
                    entities: stringList(fact.entities, 5),
                };
            })
            .filter(fact => fact.content.length >= 4)
            .slice(0, 8);
        if (summary || facts.length > 0) return { summary, facts };
    } catch {
        // Older/custom prompts may still return plain text. Keep them compatible.
    }
    return { summary: raw.trim(), facts: [] };
}

function isDuplicateMemory(
    content: string,
    embedding: number[] | undefined,
    existing: MemoryEntry[],
    kind: "session_summary" | "atomic_fact",
): boolean {
    const normalized = content.replace(/\s+/g, "").toLowerCase();
    return existing.some(entry => {
        if (entry.content.replace(/\s+/g, "").toLowerCase() === normalized) return true;
        const existingKind = String(entry.metadata?.kind ?? "session_summary");
        if (existingKind !== kind) return false;
        if (keywordOverlapRatio(entry.content, content) >= 0.9) return true;
        return Boolean(embedding && entry.embedding && cosineSimilarity(embedding, entry.embedding) >= 0.94);
    });
}

/**
 * Check if summarization should run based on event counter, then execute.
 * Trigger: counter >= summarizationEventInterval.
 * API config is resolved from Memory settings (summary may inherit the main text API).
 */
export async function maybeRunSummarization(
    characterId: string,
    characterName: string
): Promise<void> {
    const config = loadMemoryConfig();
    if (!config.autoSummarizeEnabled) return;

    const counter = getEventCounter(characterId);
    if (counter < config.summarizationEventInterval) return;

    if (summarizingSet.has(characterId)) return;
    summarizingSet.add(characterId);
    try {
        await runSummarizationPipeline(characterId, characterName);
    } finally {
        summarizingSet.delete(characterId);
    }
}

/**
 * Run the full summarization pipeline.
 * Reads events since last summarization, summarizes them, saves as long-term memory.
 * Does NOT delete short-term events — they are only trimmed by token budget elsewhere.
 * API config is resolved from auxiliary binding (global, not per-character).
 */
export async function runSummarizationPipeline(
    characterId: string,
    characterName: string,
    options?: {
        force?: boolean;
        /** 手动指定总结起点（覆盖进度水位线）；force 为真时忽略 */
        sinceTimestamp?: string;
    }
): Promise<{ success: boolean; error?: string; stored?: number; skipped?: number }> {
    const config = loadMemoryConfig();

    const apiConfig = resolveMemorySummaryApiConfig(config);
    if (!apiConfig) {
        return { success: false, error: "未配置记忆总结 API（请在记忆设置 → 记忆 API 中填写）" };
    }

    // Read native app data (chat messages, moments) directly — no separate event log
    const afterTimestamp = options?.force
        ? undefined
        : options?.sinceTimestamp ?? (getLastSummarizedTimestamp(characterId) ?? undefined);
    // 记忆来源开关同样作用于长期总结：被关掉的来源不进总结素材。
    // 进度水位线取「过滤后」最后一条的时间，因此关掉的来源不会把水位线推过头，
    // 但已被水位线越过的内容重新打开后也不会回补——这一点在设置里已注明。
    const allEntries = filterTimelineByAllowedSources(
        loadNativeTimeline(characterId, afterTimestamp ? { afterTimestamp } : undefined),
        config.shortTermAllowedSources,
    );

    if (allEntries.length < 4) {
        if (!options?.force) resetEventCounter(characterId);
        return { success: false, error: allEntries.length === 0 ? "没有可总结的事件" : "事件不足 4 条" };
    }

    const formatted = formatTimelineForSummarization(allEntries);
    if (!formatted) return { success: false, error: "格式化事件数据失败" };

    const { eventsText, earliest, latest } = formatted;

    // Use user-editable prompt template from config, with placeholder substitution
    const promptTemplate = config.summarizationPrompt?.trim() || DEFAULT_SUMMARIZATION_PROMPT;
    const summaryPrompt = promptTemplate
        .replace(/\{\{char\}\}/gi, characterName)
        .replace(/\{\{earliest\}\}/gi, earliest)
        .replace(/\{\{latest\}\}/gi, latest)
        .replace(/\{\{events\}\}/gi, eventsText)
        + (/"facts"/i.test(promptTemplate) ? "" : STRUCTURED_OUTPUT_SUFFIX);

    // Call LLM for summarization — compatible with all providers
    const result = await simpleLLMCall(
        apiConfig,
        [{ role: "user", content: summaryPrompt }],
        { temperature: 0.3 },
    );

    if (!result.content) {
        return { success: false, error: result.error || "LLM 返回了空内容" };
    }

    if (result.wasTruncated) {
        console.warn("[MemorySummarizer] Summary generation truncated:", result.finishReason);
        return { success: false, error: "记忆总结结果疑似被截断，已取消入库，请稍后重试或提高模型输出上限" };
    }

    const extraction = parseMemoryExtraction(result.content);
    const candidates = [
        ...(extraction.summary ? [{
            content: extraction.summary,
            importance: 0.72,
            tags: [] as string[],
            entities: [] as string[],
            kind: "session_summary" as const,
        }] : []),
        ...extraction.facts.map(fact => ({ ...fact, kind: "atomic_fact" as const })),
    ];
    if (candidates.length === 0) {
        return { success: false, error: "记忆总结结果为空" };
    }

    // Generate candidate embeddings in one batch when configured.
    let candidateEmbeddings: Array<number[] | undefined> = candidates.map(() => undefined);
    const embeddingApiConfig = config.vectorRecallEnabled ? resolveMemoryEmbeddingApiConfig(config) : null;
    if (embeddingApiConfig) {
        try {
            const embeddings = await generateEmbeddings(candidates.map(candidate => candidate.content), embeddingApiConfig, {
                model: config.embeddingApi.model,
            });
            if (embeddings) {
                candidateEmbeddings = embeddings;
            } else {
                candidateEmbeddings = await Promise.all(candidates.map(async candidate =>
                    await generateEmbedding(candidate.content, embeddingApiConfig, { model: config.embeddingApi.model }) ?? undefined
                ));
            }
        } catch { /* ignore */ }
    }

    // Determine sourceApp: use the most common source among summarized entries
    const sourceCounts = new Map<string, number>();
    for (const e of allEntries) {
        sourceCounts.set(e.sourceApp, (sourceCounts.get(e.sourceApp) || 0) + 1);
    }
    let dominantSource = "chat";
    let maxCount = 0;
    for (const [src, count] of sourceCounts) {
        if (count > maxCount) { dominantSource = src; maxCount = count; }
    }
    const sourceSessionIds = Array.from(new Set(
        allEntries
            .map(entry => entry.sessionId)
            .filter((sessionId): sessionId is string => Boolean(sessionId)),
    ));

    // Save summary + atomic facts with source traceability and duplicate protection.
    const now = new Date().toISOString();
    const existing = await loadMemoryEntries(characterId);
    let stored = 0;
    let skipped = 0;
    for (let index = 0; index < candidates.length; index += 1) {
        const candidate = candidates[index];
        const embedding = candidateEmbeddings[index];
        if (isDuplicateMemory(candidate.content, embedding, existing, candidate.kind)) {
            skipped += 1;
            continue;
        }
        const longTermEntry: MemoryEntry = {
            id: `mem_lt_${Date.now()}_${index}_${Math.random().toString(36).slice(2, 8)}`,
            characterId,
            sourceApp: dominantSource as MemoryEntry["sourceApp"],
            type: "long_term",
            content: candidate.content,
            embedding,
            importance: candidate.importance,
            createdAt: now,
            updatedAt: now,
            sourceMessageIds: allEntries.map(entry => entry.id),
            metadata: {
                kind: candidate.kind,
                status: "active",
                tags: candidate.tags,
                entities: candidate.entities,
                summarizedEvents: allEntries.length,
                timeSpan: `${earliest} ~ ${latest}`,
                sourceSessionIds,
                sourceEventIds: allEntries.map(entry => entry.id),
            },
        };
        await saveMemoryEntry(longTermEntry);
        existing.push(longTermEntry);
        stored += 1;
    }

    if (stored === 0 && skipped === 0) {
        return { success: false, error: "没有提取到可保存的长期记忆" };
    }

    // Update last summarized timestamp + reset counter
    setLastSummarizedTimestamp(characterId, latest);
    resetEventCounter(characterId);

    // Enforce long-term limit
    const allLongTerm = (await loadMemoryEntries(characterId)).filter(entry => entry.type === "long_term");
    if (allLongTerm.length > config.maxLongTermEntries) {
        const excess = allLongTerm.slice(0, allLongTerm.length - config.maxLongTermEntries);
        await deleteMemoryEntries(excess.map(e => e.id));
    }

    if (stored > 0) {
        incrementCoreMemoryCounter(characterId);
        await maybeRunCoreMemoryPipeline(characterId, characterName);
    }

    console.log(`[MemorySummarizer] Summarized ${allEntries.length} entries → ${stored} stored, ${skipped} skipped`);
    return { success: true, stored, skipped };
}
