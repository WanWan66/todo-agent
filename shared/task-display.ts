export const DEFAULT_APP_TIME_ZONE = "Asia/Shanghai";
export const effortLabels = { small: "小", medium: "中", large: "大" };

export function dateInTimeZone(now: Date, timeZone = DEFAULT_APP_TIME_ZONE): string {
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).formatToParts(now);
  const part = (type: Intl.DateTimeFormatPartTypes) =>
    parts.find((item) => item.type === type)?.value;
  return [part("year")?.padStart(4, "0"), part("month"), part("day")].join("-");
}

export function normalizePriority(value: unknown): { rank: number; label: string } {
  if (value === "high" || value === 1) return { rank: 3, label: "高" };
  if (value === "medium" || value === 2) return { rank: 2, label: "中" };
  if (value === "low" || value === 3) return { rank: 1, label: "低" };
  if (typeof value === "number" && Number.isFinite(value)) {
    return { rank: 0, label: String(value) };
  }
  return { rank: -1, label: "" };
}
