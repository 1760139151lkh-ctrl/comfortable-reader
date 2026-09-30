/** A deliberately bounded reader for colored ASCII PLY, not an executable format. */
export type GeometryData = {positions: Float32Array; colors: Float32Array; faces: Uint32Array; comments: string[]; min: number[]; max: number[]};
export function parsePly(bytes: ArrayBuffer): GeometryData {
  if (bytes.byteLength > 8 * 1024 * 1024) throw new Error('三维原件超过 8 MiB 交互读取上限，可继续查看参考图与原件信息。');
  const lines = new TextDecoder('utf-8', {fatal: true}).decode(bytes).replace(/\r/g, '').trim().split('\n');
  if (lines[0] !== 'ply' || lines[1] !== 'format ascii 1.0') throw new Error('此视图支持 ASCII PLY 1.0。原件仍保留，可查看本章的参考图。');
  let n = 0, f = 0, end = -1, element = '', faceProperty = false;
  const props: string[] = [], comments: string[] = [];
  for (let i = 2; i < Math.min(lines.length, 256); i++) {
    const row = lines[i].trim().split(/\s+/);
    if (row[0] === 'end_header') {end = i + 1; break;}
    if (row[0] === 'comment') {comments.push(lines[i].slice(8)); continue;}
    if (row[0] === 'obj_info') continue;
    if (row[0] === 'element') {
      const count = Number(row[2]);
      if (!Number.isInteger(count) || count < 0) throw new Error('PLY 元素数量无效。');
      element = row[1];
      if (element === 'vertex' && !n && !f) {n = count; if (n < 1 || n > 200000) throw new Error('点数超出 1—200000 的查看范围。');}
      else if (element === 'face' && n && !f) {f = count; if (f > 400000) throw new Error('面数超出查看范围。');}
      else throw new Error('PLY 包含不支持或重复的元素。');
    } else if (row[0] === 'property') {
      if (element === 'vertex' && row.length === 3 && row[1] !== 'list' && !props.includes(row[2]) && props.length < 24) props.push(row[2]);
      else if (element === 'face' && row.join(' ') === 'property list uchar int vertex_indices' && !faceProperty) faceProperty = true;
      else throw new Error('PLY 属性结构不受当前视图支持。');
    } else throw new Error('PLY 头部含未知声明。');
  }
  if (end < 0 || !n || lines.length !== end + n + f || (f && !faceProperty)) throw new Error('PLY 数据长度与声明不符。');
  const xyz = ['x', 'y', 'z'].map(k => props.indexOf(k));
  const rgb = ['red', 'green', 'blue'].map(k => props.indexOf(k));
  if (xyz.includes(-1) || (rgb.some(i => i >= 0) && rgb.includes(-1))) throw new Error('PLY 坐标或颜色列不完整。');
  const positions = new Float32Array(n * 3), colors = new Float32Array(n * 3), faces = new Uint32Array(f * 3);
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < n; i++) {
    const values = lines[end + i].trim().split(/\s+/).map(Number);
    if (values.length !== props.length || values.some(v => !Number.isFinite(v))) throw new Error(`顶点 ${i + 1} 的数值无效。`);
    for (let j = 0; j < 3; j++) {
      const v = values[xyz[j]], color = rgb[0] < 0 ? 180 : values[rgb[j]];
      if (Math.abs(v) > 1e7 || color < 0 || color > 255 || !Number.isInteger(color)) throw new Error('顶点坐标或颜色超出允许范围。');
      positions[i * 3 + j] = v; colors[i * 3 + j] = color / 255;
      min[j] = Math.min(min[j], v); max[j] = Math.max(max[j], v);
    }
  }
  for (let i = 0; i < f; i++) {
    const values = lines[end + n + i].trim().split(/\s+/).map(Number);
    if (values.length !== 4 || values[0] !== 3 || values.slice(1).some(v => !Number.isInteger(v) || v < 0 || v >= n)) throw new Error(`面 ${i + 1} 不是有效的三角面。`);
    faces.set(values.slice(1), i * 3);
  }
  return {positions, colors, faces, comments, min, max};
}
export function geometryChanges(base: GeometryData, next: GeometryData): {flags: Float32Array; count: number; maximum: number} | null {
  if (base.positions.length !== next.positions.length || base.faces.length !== next.faces.length
      || base.faces.some((v, i) => v !== next.faces[i])) return null;
  const flags = new Float32Array(base.positions.length / 3); let count = 0, maximum = 0;
  for (let i = 0; i < flags.length; i++) {
    const d = Math.hypot(...[0,1,2].map(j => next.positions[i*3+j] - base.positions[i*3+j]));
    if (d > 1e-6) {flags[i] = 1; count++; maximum = Math.max(maximum, d);}
  }
  return {flags, count, maximum};
}
