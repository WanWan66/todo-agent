import OpenAI, { APIConnectionError } from "openai";
import type {
  ChatCompletionCreateParamsNonStreaming,
  ChatCompletionMessageToolCall,
} from "openai/resources/chat/completions";

export const model = process.env.DEEPSEEK_MODEL || "deepseek-flash";

export interface ChatCompletionClientLike {
  chat: {
    completions: {
      create(request: ChatCompletionCreateParamsNonStreaming): PromiseLike<{
        choices: {
          message: { content?: string | null; tool_calls?: ChatCompletionMessageToolCall[] };
        }[];
      }>;
    };
  };
}

export function getModelClient(): OpenAI {
  const apiKey = process.env.DEEPSEEK_API_KEY;
  if (!apiKey) throw new Error("请在 .env 中设置 DEEPSEEK_API_KEY 后重启服务。");
  return new OpenAI({ apiKey, baseURL: "https://api.deepseek.com", maxRetries: 0 });
}

export function getFallbackModel(): string | undefined {
  return process.env.DEEPSEEK_FALLBACK_MODEL?.trim() || undefined;
}

type ModelErrorKind =
  | "authentication"
  | "balance"
  | "permission"
  | "rate-limit"
  | "bad-request"
  | "model-not-found"
  | "service"
  | "network"
  | "unknown";

function errorFields(value: unknown) {
  return typeof value === "object" && value !== null ? (value as Record<string, unknown>) : {};
}

export function classifyModelError(error: unknown): ModelErrorKind {
  const { status, code, message, error: body, cause } = errorFields(error);
  switch (status) {
    case 401:
      return "authentication";
    case 402:
      return "balance";
    case 403:
      return "permission";
    case 429:
      return "rate-limit";
    case 400: {
      const details = errorFields(body);
      const names = [code, details.code, message, details.message];
      return names.some(
        (value) =>
          typeof value === "string" &&
          /\bmodel_not_found\b|\bmodel\s+(?:(?:["'][^"'\r\n]+["']|[\x60][^\x60\r\n]+[\x60]|[\w]+[-/.:][\w./:-]+)\s+)?(?:(?:was|does)\s+)?not\s+(?:found|exist)\b/i.test(
            value,
          ),
      )
        ? "model-not-found"
        : "bad-request";
    }
  }
  if (typeof status === "number") return status >= 500 && status <= 599 ? "service" : "unknown";

  const fields = [code, message, errorFields(cause).code];
  if (
    error instanceof APIConnectionError ||
    fields.some(
      (value) =>
        typeof value === "string" &&
        /\b(?:ECONNRESET|ETIMEDOUT|EAI_AGAIN|ENOTFOUND|ECONNREFUSED|ENETUNREACH|UND_ERR_\w+|fetch failed|network (?:error|timeout)|(?:connection|request) (?:timed out|timeout|reset))\b/i.test(
          value,
        ),
    )
  ) {
    return "network";
  }
  return "unknown";
}

export const modelErrorMessages: Record<ModelErrorKind, string> = {
  authentication: "DeepSeek API Key 无效，请检查配置。",
  balance: "DeepSeek 账户余额不足，请充值后重试。",
  permission: "DeepSeek API Key 缺少访问权限，请检查配置。",
  "rate-limit": "DeepSeek 请求已限流，请稍后重试。",
  "bad-request": "DeepSeek 拒绝了请求，请检查模型配置或缩短对话后重试。",
  "model-not-found": "DeepSeek 找不到配置的模型，请检查主模型和备用模型名称。",
  service: "DeepSeek 服务暂时异常，请稍后重试。",
  network: "连接 DeepSeek 时发生网络异常，请稍后重试。",
  unknown: "模型请求失败，请稍后重试。",
};
