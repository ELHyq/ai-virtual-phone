import { NextResponse } from "next/server";
import { proxyFetch } from "@/lib/proxy-fetch";

export const runtime = "nodejs";
export const maxDuration = 60;

const FISH_TTS_URL = "https://api.fish.audio/v1/tts";
const DEFAULT_MODEL = "s2.1-pro-free";
const SUPPORTED_MODELS = new Set(["s2.1-pro", "s2.1-pro-free", "s2-pro", "s1"]);
const MAX_TEXT_LENGTH = 5_000;

function cleanText(value: unknown, maxLength: number): string {
    return typeof value === "string" ? value.trim().slice(0, maxLength) : "";
}

function isSameOriginRequest(request: Request): boolean {
    if (request.headers.get("sec-fetch-site") === "same-origin") return true;
    const origin = request.headers.get("origin");
    if (!origin) return process.env.NODE_ENV !== "production";
    try {
        const originHost = new URL(origin).host;
        const requestHosts = [
            request.headers.get("x-forwarded-host"),
            request.headers.get("host"),
            new URL(request.url).host,
        ].filter(Boolean);
        return requestHosts.includes(originHost);
    } catch {
        return false;
    }
}

async function upstreamError(response: Response): Promise<string> {
    const text = await response.text().catch(() => "");
    try {
        const data = JSON.parse(text) as Record<string, unknown>;
        return cleanText(data.message ?? data.error, 500) || `HTTP ${response.status}`;
    } catch {
        return cleanText(text, 500) || `HTTP ${response.status}`;
    }
}

export async function POST(request: Request) {
    if (!isSameOriginRequest(request)) {
        return NextResponse.json({ error: "forbidden", message: "仅允许当前站点调用 Fish Audio。" }, { status: 403 });
    }

    const apiKey = process.env.FISH_AUDIO_API_KEY?.trim() || "";
    if (!apiKey) {
        return NextResponse.json(
            { error: "missing_fish_audio_api_key", message: "当前部署环境尚未配置 FISH_AUDIO_API_KEY。" },
            { status: 503 },
        );
    }

    const body = await request.json().catch(() => ({})) as Record<string, unknown>;
    const text = cleanText(body.text, MAX_TEXT_LENGTH + 1);
    const referenceId = cleanText(body.referenceId, 128);
    const requestedModel = cleanText(body.model, 64) || DEFAULT_MODEL;

    if (!text || text.length > MAX_TEXT_LENGTH) {
        return NextResponse.json({ error: "invalid_text", message: `文本长度需为 1-${MAX_TEXT_LENGTH} 个字符。` }, { status: 400 });
    }
    if (!referenceId || !/^[A-Za-z0-9_-]{4,128}$/.test(referenceId)) {
        return NextResponse.json({ error: "invalid_reference_id", message: "Fish Audio Reference ID 格式不正确。" }, { status: 400 });
    }
    if (!SUPPORTED_MODELS.has(requestedModel)) {
        return NextResponse.json({ error: "invalid_model", message: `不支持的 Fish Audio 模型：${requestedModel}` }, { status: 400 });
    }

    try {
        const response = await proxyFetch(FISH_TTS_URL, {
            method: "POST",
            headers: {
                Authorization: `Bearer ${apiKey}`,
                "Content-Type": "application/json",
                model: requestedModel,
            },
            body: JSON.stringify({
                text,
                reference_id: referenceId,
                format: "mp3",
            }),
        });

        if (!response.ok) {
            return NextResponse.json(
                { error: "fish_audio_failed", message: await upstreamError(response) },
                { status: response.status >= 400 && response.status < 500 ? response.status : 502 },
            );
        }

        return new Response(response.body, {
            status: 200,
            headers: {
                "Content-Type": response.headers.get("content-type") || "audio/mpeg",
                "Cache-Control": "no-store",
            },
        });
    } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        return NextResponse.json({ error: "fish_audio_failed", message: message.slice(0, 500) }, { status: 502 });
    }
}
