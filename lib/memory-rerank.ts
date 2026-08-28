// Optional OpenAI-compatible rerank step for memory retrieval.

import { buildRequestHeaders, determineBaseUrl } from "./api-helpers";
import type { MemoryEntry } from "./memory-types";
import type { ApiConfig } from "./settings-types";

type RerankResponseItem = {
    index?: unknown;
    relevance_score?: unknown;
    score?: unknown;
};

/**
 * Rerank a fused candidate pool. Supports the common /rerank contract used by
 * Cohere-compatible services, Jina and SiliconFlow. Returns null when the
 * optional service is unavailable so callers can retain their RRF ordering.
 */
export async function rerankMemoryEntries(
    query: string,
    candidates: MemoryEntry[],
    apiConfig: ApiConfig,
): Promise<MemoryEntry[] | null> {
    if (candidates.length === 0) return [];
    const model = apiConfig.defaultModel?.trim();
    if (!model) return null;

    const baseUrl = determineBaseUrl(apiConfig).replace(/\/$/, "");
    if (!baseUrl) return null;
    const url = baseUrl.endsWith("/rerank") ? baseUrl : `${baseUrl}/rerank`;

    try {
        const headers = buildRequestHeaders(apiConfig, baseUrl);
        if (!apiConfig.apiKey) {
            delete headers.Authorization;
            delete headers["x-api-key"];
        }
        const response = await fetch(url, {
            method: "POST",
            headers,
            body: JSON.stringify({
                model,
                query,
                documents: candidates.map(candidate => candidate.content),
                top_n: candidates.length,
            }),
        });
        if (!response.ok) {
            console.warn(`[MemoryRerank] API error ${response.status}: ${await response.text()}`);
            return null;
        }

        const payload = await response.json();
        const rows: RerankResponseItem[] = Array.isArray(payload?.results)
            ? payload.results
            : Array.isArray(payload?.data)
                ? payload.data
                : [];
        if (rows.length === 0) return null;

        const ranked: MemoryEntry[] = [];
        const seen = new Set<number>();
        for (const row of rows) {
            const index = Number(row.index);
            if (!Number.isInteger(index) || index < 0 || index >= candidates.length || seen.has(index)) continue;
            ranked.push(candidates[index]);
            seen.add(index);
        }
        if (ranked.length === 0) return null;

        // Some providers return fewer rows than requested; keep the remaining
        // candidates in their original RRF order instead of dropping memory.
        candidates.forEach((candidate, index) => {
            if (!seen.has(index)) ranked.push(candidate);
        });
        return ranked;
    } catch (error) {
        console.warn("[MemoryRerank] fetch error:", error);
        return null;
    }
}
