type OpenLink = (url: string) => void;

// Render only text and explicitly supported markup; HTML is always literal.
export function appendInline(parent: HTMLElement, text: string, openLink: OpenLink): void {
  const pattern = /`([^`\n]+)`|\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)|\*\*([^*\n]+)\*\*|(?<![\w\\])\*([^*\n]+)\*/g;
  let end = 0;
  for (const match of text.matchAll(pattern)) {
    parent.append(document.createTextNode(text.slice(end, match.index)));
    const node = document.createElement(match[1] ? 'code' : match[2] ? 'a' : match[4] ? 'strong' : 'em');
    node.textContent = match[1] ?? match[2] ?? match[4] ?? match[5];
    if (match[2]) {
      (node as HTMLAnchorElement).href = match[3];
      node.addEventListener('click', event => { event.preventDefault(); openLink(match[3]); });
    }
    parent.append(node);
    end = match.index! + match[0].length;
  }
  parent.append(document.createTextNode(text.slice(end)));
}

export function renderReadableText(parent: HTMLElement, text: string, openLink: OpenLink, markdown = true): void {
  parent.classList.add('study-prose');
  const lines = text.replace(/\r\n?/g, '\n').split('\n');
  let paragraph: string[] = [];
  const flush = () => {
    if (!paragraph.length) return;
    const p = document.createElement('p');
    if (markdown) appendInline(p, paragraph.join('\n'), openLink); else p.textContent = paragraph.join('\n');
    parent.append(p); paragraph = [];
  };
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (!line.trim()) { flush(); continue; }
    const fence = markdown && line.match(/^\s*(`{3,}|~{3,})/);
    if (fence) {
      flush(); const code: string[] = []; const marker = fence[1][0];
      while (++i < lines.length && !new RegExp('^\\s*' + marker + '{3,}\\s*$').test(lines[i])) code.push(lines[i]);
      const pre = document.createElement('pre'); pre.className = 'study-code-readonly'; pre.textContent = code.join('\n'); parent.append(pre); continue;
    }
    const heading = markdown && line.match(/^(#{1,6})\s+(.+)$/);
    if (heading) { flush(); const h = document.createElement('h' + Math.min(heading[1].length + 1, 6)); appendInline(h, heading[2], openLink); parent.append(h); continue; }
    if(markdown&&line.includes('|')&&i+1<lines.length&&/^\s*\|?\s*:?-{3,}/.test(lines[i+1])){
      flush();const wrapper=document.createElement('div');wrapper.className='study-data-scroll';wrapper.tabIndex=0;wrapper.setAttribute('aria-label','资料表格，可左右滚动');const table=document.createElement('table');table.className='study-table study-markdown-table';
      const cells=(value:string)=>value.trim().replace(/^\|/,'').replace(/\|$/,'').split(/(?<!\\)\|/).map(cell=>cell.trim().replace(/\\\|/g,'|'));
      const head=table.createTHead().insertRow();for(const value of cells(line)){const th=document.createElement('th');th.scope='col';appendInline(th,value,openLink);head.append(th);}i++;
      const body=table.createTBody();while(i+1<lines.length&&lines[i+1].includes('|')&&lines[i+1].trim()){const row=body.insertRow();for(const value of cells(lines[++i])){const td=row.insertCell();appendInline(td,value,openLink);}}
      wrapper.append(table);parent.append(wrapper);continue;
    }
    const item = markdown && line.match(/^(\s*)(?:([-+*])|\d+[.)])\s+(.+)$/);
    if (item) {
      flush(); const list = document.createElement(item[2] ? 'ul' : 'ol');
      do {
        const current = lines[i].match(/^\s*(?:[-+*]|\d+[.)])\s+(.+)$/)!;
        const li = document.createElement('li'); appendInline(li, current[1], openLink); list.append(li);
      } while (i + 1 < lines.length && /^(?:\s*)(?:[-+*]|\d+[.)])\s+/.test(lines[i + 1]) && ++i);
      parent.append(list); continue;
    }
    if (markdown && /^>\s?/.test(line)) { flush(); const quote = document.createElement('blockquote'); appendInline(quote, line.replace(/^>\s?/, ''), openLink); parent.append(quote); continue; }
    if (markdown && /^\s*(?:-{3,}|\*{3,})\s*$/.test(line)) { flush(); parent.append(document.createElement('hr')); continue; }
    paragraph.push(line);
  }
  flush();
}

export function parseDelimited(text: string, delimiter = ','): string[][] {
  const rows: string[][] = []; let row: string[] = [], field = '', quoted = false;
  text = text.replace(/^\uFEFF/, '');
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    if (char === '"') {
      if (quoted && text[i + 1] === '"') { field += '"'; i++; }
      else if (quoted || field === '') quoted = !quoted;
      else field += char;
    } else if (!quoted && char === delimiter) { row.push(field); field = ''; }
    else if (!quoted && (char === '\n' || char === '\r')) {
      if (char === '\r' && text[i + 1] === '\n') i++;
      row.push(field); rows.push(row); row = []; field = '';
    } else field += char;
  }
  if (quoted) throw new Error('表格的引号没有闭合，原始文本仍保留');
  if (field || row.length) { row.push(field); rows.push(row); }
  return rows;
}

export function renderDelimited(parent: HTMLElement, text: string, delimiter = ','): void {
  const rows = parseDelimited(text, delimiter);
  const hint = document.createElement('p'); hint.className = 'study-scroll-hint'; hint.textContent = '按字段对照阅读；宽表可左右滚动。原始记录完整保留。';
  const wrapper = document.createElement('div'); wrapper.className = 'study-data-scroll'; wrapper.tabIndex = 0; wrapper.setAttribute('aria-label', '完整数据表，可上下和左右滚动');
  const table = document.createElement('table'); table.className = 'study-table study-data-table';
  rows.forEach((cells, index) => {
    const section = index === 0 ? (table.tHead ?? table.createTHead()) : (table.tBodies[0] ?? table.createTBody());
    const tr = section.insertRow();
    for (const cell of cells) { const td = document.createElement(index === 0 ? 'th' : 'td'); td.textContent = cell; if (index === 0) td.setAttribute('scope', 'col'); tr.append(td); }
  });
  wrapper.append(table); parent.append(hint, wrapper);
  const detail = document.createElement('details'); detail.className = 'study-details';
  const summary = document.createElement('summary'); summary.textContent = '查看完整原始文本';
  const raw = document.createElement('pre'); raw.className = 'study-code-readonly'; raw.textContent = text;
  detail.append(summary, raw); parent.append(detail);
}
