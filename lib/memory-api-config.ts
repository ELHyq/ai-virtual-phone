import type { ApiConfig } from "./settings-types";
import type { MemoryApiConnection, MemoryConfig } from "./memory-types";
import { loadApiConfigs, loadBindingConfig } from "./settings-storage";

export const MEMORY_API_PROVIDERS = [
    "OpenAI",
    "Anthropic",
    "Google",
    "DeepSeek",
    "Groq",
    "OpenRouter",
    "Moonshot",
    "Zhipu",
    "SiliconFlow",
    "TogetherAI",
    "Custom",
] as const;

export function memoryConnectionFromApiConfig(
    apiConfig: ApiConfig,
    mode: MemoryApiConnection["mode"] = "custom",
): MemoryApiConnection {
    return {
        mode,
        provider: apiConfig.provider || "Custom",
        baseUrl: apiConfig.baseUrl ?? "",
        apiKey: apiConfig.apiKey ?? "",
        model: apiConfig.defaultModel ?? "",
    };
}

export function memoryConnectionToApiConfig(
    connection: MemoryApiConnection,
    id: string,
    name: string,
): ApiConfig | null {
    const model = connection.model.trim();
    const baseUrl = connection.baseUrl.trim();
    if (!model || (connection.provider === "Custom" && !baseUrl)) return null;
    return {
        id,
        name,
        provider: connection.provider || "Custom",
        apiKey: connection.apiKey,
        baseUrl: baseUrl || undefined,
        defaultModel: model,
        enableNativeTools: false,
        enableImageRecognition: false,
        enableImageGeneration: false,
        preventEmptyGenerateRambling: true,
    };
}

export function resolveMainApiConfig(): ApiConfig | null {
    const binding = loadBindingConfig();
    const configs = loadApiConfigs();
    if (binding.globalDefaults.apiConfigId) {
        const selected = configs.find(config => config.id === binding.globalDefaults.apiConfigId);
        if (selected) return selected;
    }
    return configs[0] ?? null;
}

export function resolveMemorySummaryApiConfig(config: MemoryConfig): ApiConfig | null {
    if (config.summaryApi.mode === "inherit_main") return resolveMainApiConfig();
    return memoryConnectionToApiConfig(config.summaryApi, "memory-summary-direct", "记忆总结 API");
}

export function resolveMemoryEmbeddingApiConfig(config: MemoryConfig): ApiConfig | null {
    return memoryConnectionToApiConfig(config.embeddingApi, "memory-embedding-direct", "Embedding API");
}

export function resolveMemoryRerankApiConfig(config: MemoryConfig): ApiConfig | null {
    return memoryConnectionToApiConfig(config.rerankApi, "memory-rerank-direct", "Rerank API");
}
