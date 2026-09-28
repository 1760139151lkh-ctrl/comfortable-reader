export type ReadingMetric = {
  kind: 'body' | 'reference' | 'position' | 'page';
  position: number;
  total: number;
  percent: number | null;
};

export function readingMetricLabel(metric: ReadingMetric | null | undefined): {text: string; percent: number | null} {
  if (!metric) return {text: '已保存阅读位置', percent: null};
  if (metric.kind === 'reference') return {text: '配套资料 · 不计入正文进度', percent: null};
  if (!['body', 'position', 'page'].includes(metric.kind) || !Number.isFinite(metric.position)
      || !Number.isFinite(metric.total) || metric.total < 1) return {text: '已保存阅读位置', percent: null};
  const position = Math.min(Math.round(metric.total), Math.max(1, Math.round(metric.position)));
  const percent = Number.isFinite(metric.percent) && metric.percent !== null ? Math.max(0, Math.min(100, Math.round(metric.percent * 100))) : null;
  const label = metric.kind === 'body' ? '正文位置' : metric.kind === 'page' ? '页' : '阅读位置';
  return {text: `${label} ${position} / ${Math.round(metric.total)}${percent === null ? '' : ` · ${percent}%`}`, percent};
}
