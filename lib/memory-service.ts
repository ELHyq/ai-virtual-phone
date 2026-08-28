// lib/memory-service.ts
// High-level memory orchestration: retrieve long-term memories for prompt injection.

import type { MemoryConfig, MemoryEntry } from "./memory-types";
import { loadMemoryEntriesByType } from "./memory-storage";
import { resolveMemoryEmbeddingApiConfig, resolveMemoryRerankApiConfig } from "./memory-api-config";
import { generateEmbedding, cosineSimilarity, keywordSearch } from "./memory-embedding";
import { estimateTokens } from "./token-counter";
import { rerankMemoryEntries } from "./memory-rerank";

/**
 * Retrieve relevant long-term memories for prompt injection.
 * Strategy: always rank active memories, then cap by both top-K and token budget.
 * Keyword, vector, entity/tag, recency and importance signals are fused with RRF.
 * Embedding and rerank APIs are resolved directly from Memory settings.
 */
export async function retrieveMemoriesForPrompt(
    characterId: string,
    currentContext: string,
    config: MemoryConfig
): Promise<MemoryEntry[]> {
    const longTermEntries = (await loadMemoryEntriesByType(characterId, "long_term"))
        .filter(isMemoryActive);
    if (longTermEntries.length === 0 || !currentContext.trim()) return [];

    const topK = Math.min(30, Math.max(1, config.recallTopK || 10));
    const rankedLists: Array<{ entries: MemoryEntry[]; weight: number }> = [];

    const keywordRank = keywordSearch(currentContext, longTermEntries, longTermEntries.length)
        .map(result => result.entry);
    if (keywordRank.length > 0) rankedLists.push({ entries: keywordRank, weight: 1 });

    const entityRank = rankByMetadataMatch(currentContext, longTermEntries);
    if (entityRank.length > 0) rankedLists.push({ entries: entityRank, weight: 0.8 });

    const embeddingApiConfig = config.vectorRecallEnabled ? resolveMemoryEmbeddingApiConfig(config) : null;
    if (embeddingApiConfig) {
        const queryEmbedding = await generateEmbedding(currentContext, embeddingApiConfig, {
            model: config.embeddingApi.model,
        });
        if (queryEmbedding) {
            const withEmbeddings = longTermEntries.filter(m => m.embedding && m.embedding.length > 0);
            if (withEmbeddings.length > 0) {
                const scored = withEmbeddings.map(entry => ({
                    entry,
                    score: cosineSimilarity(queryEmbedding, entry.embedding!),
                }));
                scored.sort((a, b) => b.score - a.score);
                rankedLists.push({ entries: scored.map(s => s.entry), weight: 1 });
            }
        }
    }

    const recentRank = [...longTermEntries].sort(
        (a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
    );
    rankedLists.push({ entries: recentRank, weight: 0.2 });

    const fused = fuseRankings(rankedLists, longTermEntries);
    let ranked = fused;
    const rerankApiConfig = config.rerankEnabled !== false
        ? resolveMemoryRerankApiConfig(config)
        : null;
    if (rerankApiConfig) {
        const candidatePoolSize = Math.min(fused.length, Math.max(topK * 4, 20));
        const reranked = await rerankMemoryEntries(
            currentContext,
            fused.slice(0, candidatePoolSize),
            rerankApiConfig,
        );
        if (reranked) ranked = reranked;
    }
    return fillByBudget(ranked.slice(0, topK), config.longTermTokenBudget);
}

export async function retrieveCoreMemoriesForPrompt(
    characterId: string,
    config: MemoryConfig,
): Promise<MemoryEntry[]> {
    const coreEntries = (await loadMemoryEntriesByType(characterId, "core"))
        .filter(isMemoryActive);
    if (coreEntries.length === 0) return [];

    const sorted = [...coreEntries].sort((a, b) => {
        const aActive = a.metadata?.active ? 1 : 0;
        const bActive = b.metadata?.active ? 1 : 0;
        if (aActive !== bActive) return bActive - aActive;
        const aDate = String(a.metadata?.eventDate ?? a.updatedAt ?? a.createdAt);
        const bDate = String(b.metadata?.eventDate ?? b.updatedAt ?? b.createdAt);
        return bDate.localeCompare(aDate);
    });

    return fillByBudget(sorted, config.coreMemoryTokenBudget);
}

export function isMemoryActive(entry: MemoryEntry): boolean {
    const status = String(entry.metadata?.status ?? "active");
    return entry.metadata?.active !== false && status !== "archived" && status !== "superseded";
}

function metadataTerms(entry: MemoryEntry): string[] {
    const tags = Array.isArray(entry.metadata?.tags) ? entry.metadata.tags.map(String) : [];
    const entities = Array.isArray(entry.metadata?.entities) ? entry.metadata.entities.map(String) : [];
    return [...tags, ...entities].map(value => value.trim().toLowerCase()).filter(Boolean);
}

function rankByMetadataMatch(query: string, entries: MemoryEntry[]): MemoryEntry[] {
    const normalized = query.toLowerCase();
    return entries
        .map(entry => ({
            entry,
            score: metadataTerms(entry).reduce((score, term) => score + (normalized.includes(term) ? 1 : 0), 0),
        }))
        .filter(item => item.score > 0)
        .sort((a, b) => b.score - a.score)
        .map(item => item.entry);
}

function fuseRankings(
    lists: Array<{ entries: MemoryEntry[]; weight: number }>,
    allEntries: MemoryEntry[],
): MemoryEntry[] {
    const scores = new Map<string, number>();
    const byId = new Map(allEntries.map(entry => [entry.id, entry]));
    const RRF_K = 60;
    for (const list of lists) {
        list.entries.forEach((entry, index) => {
            scores.set(entry.id, (scores.get(entry.id) ?? 0) + list.weight / (RRF_K + index + 1));
        });
    }
    for (const entry of allEntries) {
        const importance = Number.isFinite(entry.importance) ? Math.max(0, Math.min(1, entry.importance)) : 0.5;
        scores.set(entry.id, (scores.get(entry.id) ?? 0) + importance * 0.003);
    }
    return [...scores.entries()]
        .sort((a, b) => b[1] - a[1])
        .map(([id]) => byId.get(id))
        .filter((entry): entry is MemoryEntry => Boolean(entry));
}

/** Pick entries in order until token budget is exhausted. */
function fillByBudget(entries: MemoryEntry[], budget: number): MemoryEntry[] {
    const result: MemoryEntry[] = [];
    let used = 0;
    for (const entry of entries) {
        const tokens = estimateTokens(entry.content) + 4;
        if (used + tokens > budget) break;
        result.push(entry);
        used += tokens;
    }
    return result;
}
