import { invoke, getCurrentWindow, open, isDesktop } from "./platform";
import { preparePortableBook, portableRequest, readerPackageUrl, rewritePortableResources, portableBooks, type CatalogRecord } from './portable-books';
import {initCatalogue,showCatalogue,openWebLink} from './catalogue';
import { unfinishedNoteIdentity } from './personal-records';
import { propagateBookOpenFailure, propagateDisplayFailure, ReaderContentLoadError, readableError } from './reader-errors';
import ePub, {
  Book as EpubBook,
  Contents as EpubContents,
  Location as EpubLocation,
  Rendition,
} from "epubjs";
import "./styles.css";
import "./workspace.css";
import {ReadingWorkspace} from './workspace';
import {MAX_OPEN_BOOKS,type WorkspaceState} from './workspace-layout';
import { readingMetricLabel } from './reading-progress';
import { initLearning, prepareLearning, wireLearningDocument, refreshLearningButton, learningIsOpen, learningReadingProgress, learningCfiFromBodyPosition, flushLearningBeforeClose, closeLearning, holdLearning, learningMemoryBackup } from "./learning";
import { mountImage } from './image-viewer';
import { sourceStrongPatterns, restoreSourceStrong, fitInlineStops, protectHeadingWords } from './book-typography';

type ThemeName = "paper" | "light" | "night" | "contrast";
let readerReady = false;
type AnnotationTool = "read" | "text" | "pen" | "eraser";

interface AnnotationTextStyle {
  color: string;
  fontFamily: string;
  fontSize: number;
  highlight: boolean;
  bold: boolean;
  underline: boolean;
  wave: boolean;
  box: boolean;
}

interface TextAnnotation extends AnnotationTextStyle {
  id: string;
  kind: "text-mark";
  cfiRange: string;
  selectedText: string;
  comment: string;
  createdAt: number;
}

interface FreeTextAnnotation extends AnnotationTextStyle {
  contentPlacement?: {mode:'paged'|'scroll';anchorX:number;anchorY:number;width:number;height:number};
  id: string;
  kind: "free-text";
  anchorCfi: string | null;
  progression: number;
  anchorVersion?: number;
  x: number;
  y: number;
  text: string;
  createdAt: number;
}

interface DrawingPoint {
  x: number;
  y: number;
}

interface DrawingAnnotation {
  contentPlacement?: FreeTextAnnotation['contentPlacement'];
  id: string;
  kind: "drawing";
  anchorCfi: string | null;
  progression: number;
  anchorVersion?: number;
  color: string;
  strokeWidth: number;
  points: DrawingPoint[];
  inkGeometry?: { columnWidth: number; pageHeight: number; group: string; basis: "creation" | "legacy-current-layout" };
  createdAt: number;
}

type ReaderAnnotation = TextAnnotation | FreeTextAnnotation | DrawingAnnotation;

interface PendingSelection {
  cfiRange: string;
  text: string;
  contents: EpubContents;
}

interface BookRecord {
  modifiedAt?:number|null;
  available?:boolean|null;
  bookUuid?: string | null;
  catalogSource?: CatalogRecord['catalogSource'] | null;
  id: string;
  title: string;
  author: string;
  path: string;
  addedAt: number;
  lazyPages?: number | null;
}

interface BookProgress {
  readingPreferences?: {fontScale?:number;readerFont?:"serif"|"sans"|"publisher";lineHeight?:number|null;contentWidth?:number|null} | null;
  bookUuid?:string|null;
  contentDigest?: string | null;
  readingMode?: 'paged' | 'scroll' | null;
  readingMetric?: import('./reading-progress').ReadingMetric | null;
  sourceSha256?: string | null;
  cfi: string | null;
  page: number;
  totalPages: number;
  percent: number;
  updatedAt: number;
  /** 0 = automatic, 1..10 = fixed visible page count. */
  pageMode: number;
  /** Persistent text marks, free notes, and doodles belonging to this book. */
  annotations: ReaderAnnotation[];
}

interface ReaderSession {
  workspace?: WorkspaceState | null;
  lineHeight?: number | null;
  contentWidth?: number | null;
  paneCount: number;
  paneBookIds: Array<string | null>;
  activePane: number;
  theme: ThemeName;
  fontScale: number;
  readerFont: "serif" | "sans" | "publisher";
}

interface ScanIssue {
  path: string;
  message: string;
}

interface ScanReport {
  rootsScanned: number;
  missingRoots: number;
  filesSeen: number;
  epubCandidates: number;
  loadedCandidates: number;
  booksLoaded: number;
  added: number;
  updated: number;
  duplicates: number;
  unreadable: number;
  otherBookFiles: number;
  ignoredTrees: number;
  issues: ScanIssue[];
}

interface AppSnapshot {
  books: BookRecord[];
  progress: Record<string, BookProgress>;
  session: ReaderSession;
  libraryRoots: string[];
  storagePath: string;
  scan: ScanReport;
}

interface PaneRuntime {
  openedSourceKey?:string|null;
  streamed: boolean;
  readingMode: 'paged' | 'scroll';
  opening: Promise<void> | null;
  finishOpening: (()=>void) | null;
  bookId: string | null;
  book: EpubBook | null;
  rendition: Rendition | null;
  resizeObserver: ResizeObserver | null;
  resizeTimer: number | null;
  resizeFrame: number | null;
  resizeActive: boolean;
  resizeAnchorCfi: string | null;
  /** Intentional content target; repeated layout changes must not round it again. */
  layoutAnchorCfi: string | null;
  /** Keep that target while layout changes still leave its text in view. */
  preserveSemanticFocus: boolean;
  pendingStageWidth: number;
  pendingStageHeight: number;
  generation: number;
  currentPage: number;
  endPage: number;
  totalPages: number;
  percent: number;
  cfi: string | null;
  pageMode: number;
  actualPageCount: number;
  effectiveFontScale: number;
  layoutWidth: number;
  layoutHeight: number;
  viewportWidth: number;
  viewportHeight: number;
  viewportScale: number;
  restoringLocation: boolean;
  paginationGeneration: number;
  paginationTimer: number | null;
  atomicFitFrame: number | null;
  lazyPageCount: number;
  lazyPageUrls: Map<number, string>;
  lazyPageLoading: Map<number, Promise<string>>;
  lazyPageGeneration: number;
}

interface DrawingDraft {
  paneIndex: number;
  pointerId: number;
  annotation: DrawingAnnotation;
}

interface NoteDragState {
  paneIndex: number;
  annotationId: string;
  pointerId: number;
  offsetX: number;
  offsetY: number;
}

interface InternalLayout {
  name: string;
  settings: { direction?: string };
  width: number;
  height: number;
  spreadWidth: number;
  pageWidth: number;
  delta: number;
  columnWidth: number;
  gap: number;
  divisor: number;
  props: Record<string, string | number | boolean>;
  calculate: (width: number, height: number, gap?: number) => void;
  count: (totalLength: number, pageLength?: number) => { spreads: number; pages: number };
  update: (props: Record<string, string | number | boolean>) => void;
}

type InternalRendition = Rendition & {
  _layout: InternalLayout;
  manager: {
    updateLayout: () => void;
    container?: HTMLElement;
    views?: {
      last: () => { section?: { next: () => unknown } } | undefined;
    };
  };
};

type InternalSpine = EpubBook["spine"] & {
  spineItems: Array<{ linear?: boolean }>;
};

const MAX_PANES = MAX_OPEN_BOOKS;
let workspace:ReadingWorkspace;
let toolsVisible=false;
function requireElement<T extends Element>(root: ParentNode, selector: string): T {
  const element = root.querySelector<T>(selector);
  if (!element) throw new Error(`界面元素不存在：${selector}`);
  return element;
}
const app = requireElement<HTMLElement>(document, "#app");

const icons: Record<string, string> = {
  library:
    '<svg viewBox="0 0 24 24"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20V4H6.5A2.5 2.5 0 0 0 4 6.5v13Z"/><path d="M8 7h8M8 10h6M6.5 17A2.5 2.5 0 0 0 4 19.5 2.5 2.5 0 0 0 6.5 22H20v-5Z"/></svg>',
  back: '<svg viewBox="0 0 24 24"><path d="m15 18-6-6 6-6"/></svg>',
  forward: '<svg viewBox="0 0 24 24"><path d="m9 18 6-6-6-6"/></svg>',
  groupBack:
    '<svg viewBox="0 0 24 24"><path d="m12 18-6-6 6-6"/><path d="m18 18-6-6 6-6"/></svg>',
  groupForward:
    '<svg viewBox="0 0 24 24"><path d="m6 18 6-6-6-6"/><path d="m12 18 6-6-6-6"/></svg>',
  jump:
    '<svg viewBox="0 0 24 24"><path d="M4 5h16v14H4z"/><path d="M8 9h8M8 12h5M8 15h8"/><path d="m17 12 3 3-3 3"/></svg>',
  plus: '<svg viewBox="0 0 24 24"><path d="M12 5v14M5 12h14"/></svg>',
  folder:
    '<svg viewBox="0 0 24 24"><path d="M3 6.5h6l2 2h10v9.5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V6.5Z"/></svg>',
  refresh:
    '<svg viewBox="0 0 24 24"><path d="M20 6v5h-5"/><path d="M18.2 16.4A8 8 0 1 1 19.5 9L20 11"/></svg>',
  close: '<svg viewBox="0 0 24 24"><path d="m7 7 10 10M17 7 7 17"/></svg>',
  moon:
    '<svg viewBox="0 0 24 24"><path d="M20.5 14.7A8.5 8.5 0 0 1 9.3 3.5 8.5 8.5 0 1 0 20.5 14.7Z"/></svg>',
  search:
    '<svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/></svg>',
  book:
    '<svg viewBox="0 0 24 24"><path d="M5 4.5A2.5 2.5 0 0 1 7.5 2H20v17H7.5A2.5 2.5 0 0 0 5 21.5v-17Z"/><path d="M5 21.5A2.5 2.5 0 0 1 7.5 19H20v3H7.5A2.5 2.5 0 0 1 5 19.5"/></svg>',
  check:
    '<svg viewBox="0 0 24 24"><path d="m5 12 4 4L19 6"/></svg>',
  fullscreen:
    '<svg viewBox="0 0 24 24"><path d="M8 3H3v5M16 3h5v5M21 16v5h-5M3 16v5h5"/></svg>',
  note:
    '<svg viewBox="0 0 24 24"><path d="M4 19.5V5.8A1.8 1.8 0 0 1 5.8 4h8.8L20 9.4v10.1a.5.5 0 0 1-.5.5h-14a1.5 1.5 0 0 1-1.5-1.5Z"/><path d="M14 4v6h6M8 14h8M8 17h5"/></svg>',
  pen:
    '<svg viewBox="0 0 24 24"><path d="m4 20 4.8-1.1L19 8.7a2.4 2.4 0 0 0-3.4-3.4L5.4 15.5 4 20Z"/><path d="m13.8 7.1 3.1 3.1"/></svg>',
  undo:
    '<svg viewBox="0 0 24 24"><path d="M9 7 4 12l5 5"/><path d="M5 12h8a6 6 0 0 1 6 6"/></svg>',
  trash:
    '<svg viewBox="0 0 24 24"><path d="M4 7h16M9 7V4h6v3M7 7l1 13h8l1-13M10 11v5M14 11v5"/></svg>',
};

app.innerHTML = `
  <div class="app-shell" data-theme="paper">

    <nav class="workspace-strip" aria-label="阅读空间">
      <button class="workspace-library" type="button" title="书库（Ctrl+L）">${icons.library}<span>书库</span></button>
      <button class="tools-toggle" type="button" aria-label="打开阅读工具" aria-expanded="false" title="阅读工具（F8）">${icons.book}<span>阅读工具</span></button>
      <div class="workspace-tabs" aria-label="仍然打开的书籍"></div>
      <span class="workspace-space-status" hidden></span>
      <button class="workspace-focus" type="button" hidden>专注一本</button>
      <button class="workspace-toggle" type="button" aria-expanded="false"><span>书籍 · 0</span><span aria-hidden="true">⌄</span></button>
    </nav>
    <section class="workspace-panel floating-panel" aria-label="打开的书籍与布局" aria-hidden="true" inert>
      <header><strong>阅读现场</strong><button class="icon-button workspace-panel-close" aria-label="收起阅读现场">${icons.close}</button></header>
      <div class="workspace-books"></div>
      <button class="text-action workspace-organize" type="button">重新整理布局</button>
      <p class="settings-help">拖动书名左侧的手柄摆放书籍；拖动细分隔线调整比例。窗口变小时保留原排列。</p>
      <details class="workspace-move-controls"><summary>用按钮调整位置</summary><p>移动：<strong class="workspace-current-title"></strong></p><label>放到哪本书旁边<select class="workspace-target"></select></label><div class="workspace-placement-buttons">${Object.entries({left:'左侧',right:'右侧',top:'上方',bottom:'下方',swap:'交换'}).map(([value,label])=>`<button type="button" data-place="${value}">${label}</button>`).join('')}</div></details>
    </section>
    <header class="top-toolbar" aria-label="阅读工具栏" aria-hidden="true" inert>
      <button class="icon-button library-toggle" type="button" title="书库（Ctrl+L）" aria-label="打开书库" aria-expanded="false">${icons.library}</button>
      <button class="icon-button contents-toggle" type="button" title="本书目录（Ctrl+T）" aria-label="打开本书目录" aria-expanded="false">${icons.jump}</button>
      <button class="icon-button book-search-toggle" type="button" title="书内搜索（Ctrl+F）" aria-label="搜索本书" aria-expanded="false">${icons.search}</button>
      <button class="icon-button reading-back" type="button" title="返回跳转前的位置（Alt+←）" aria-label="返回跳转前的位置" disabled>${icons.undo}</button>
      <span class="toolbar-separator"></span>
      <button class="icon-button nav-back" type="button" title="上一栏（←）" aria-label="上一页">${icons.back}</button>
      <button class="icon-button nav-forward" type="button" title="下一栏（→）" aria-label="下一页">${icons.forward}</button>
      <button class="icon-button nav-group-back" type="button" title="上一屏（PageUp）" aria-label="上一屏">${icons.groupBack}</button>
      <button class="icon-button nav-group-forward" type="button" title="下一屏（PageDown）" aria-label="下一屏">${icons.groupForward}</button>
      <div class="active-book-title">尚未打开书籍</div>
      <button class="text-button reading-settings-toggle" type="button" title="字体与阅读设置" aria-label="打开阅读设置" aria-expanded="false">Aa</button>
      <button class="icon-button theme-cycle" type="button" title="切换阅读主题" aria-label="切换阅读主题">${icons.moon}</button>
      <button class="icon-button annotation-toggle" type="button" title="笔记与原文标记（Ctrl+N）" aria-label="打开笔记与标记工具" aria-expanded="false">${icons.note}</button>
      <button class="icon-button fullscreen-toggle" type="button" title="全屏（F11）" aria-label="切换全屏">${icons.fullscreen}</button>
      <button class="icon-button tools-close" type="button" title="回到安静阅读（Escape）" aria-label="收起阅读工具">${icons.close}</button>
    </header>
    <section class="reading-settings floating-panel" aria-label="阅读设置" aria-hidden="true" inert>
      <header><strong>这本书的排版</strong><button type="button" class="icon-button settings-close" aria-label="关闭阅读设置">${icons.close}</button></header>
      <p class="settings-book-title"></p>
      <label class="setting-row"><span>这本书怎样读</span><select class="reading-mode" aria-label="阅读方式"><option value="paged">翻页 · 可选多栏</option><option value="scroll">连续滚动</option></select></label>
      <label class="setting-row"><span>这本书同屏页数</span><select class="settings-page-mode" aria-label="这本书同屏页数"><option value="0">自动 · 舒适行宽</option>${Array.from({length:10},(_,i)=>`<option value="${i+1}">${i+1} 页</option>`).join('')}</select></label>
      <label class="setting-row"><span>正文字体</span><select class="reader-font" aria-label="正文字体"><option value="serif">衬线 · 书籍</option><option value="sans">无衬线 · 清晰</option><option value="publisher">书籍原有字体</option></select></label>
      <div class="setting-row"><span>阅读字号</span><div class="font-controls"><button type="button" class="text-button font-down" title="缩小字号">A−</button><span class="font-scale-label">100%</span><button type="button" class="text-button font-up" title="放大字号">A+</button></div></div>
      <label class="setting-row"><span>行距</span><input class="reader-line-height" type="range" min="1.35" max="2.15" step="0.05" aria-label="正文行距"/></label>
      <label class="setting-row"><span>连续阅读宽度</span><select class="reader-content-width" aria-label="连续阅读宽度"><option value="34">紧凑</option><option value="44">标准</option><option value="56">舒展</option></select></label>
      <p class="settings-help">排版只影响当前书。左右键移动一页，PageUp / PageDown 移动一屏。要对照其他书，从书库打开即可。</p>
      <button class="text-action jump-toggle" type="button">跳转到阅读位置…</button>
    </section>
    <aside class="book-navigation floating-panel" aria-label="本书导航" aria-hidden="true" inert>
      <header><strong class="navigation-heading">本书目录</strong><button type="button" class="icon-button navigation-close" aria-label="收起本书导航">${icons.close}</button></header>
      <div class="navigation-tabs" role="group" aria-label="本书导航方式"><button type="button" data-navigation-tab="contents" aria-pressed="true">目录</button><button type="button" data-navigation-tab="search" aria-pressed="false">搜索</button></div>
      <form class="book-search-form" hidden><label><span class="sr-only">搜索本书内容</span><input type="search" class="book-search-input" placeholder="输入词语或短句" autocomplete="off" /></label><button type="submit">搜索</button></form>
      <p class="navigation-status" role="status"></p>
      <nav class="chapter-list" aria-label="章节目录"></nav>
      <div class="book-search-results" hidden></div>
    </aside>

    <section class="annotation-panel" aria-label="笔记与原文标记工具" aria-hidden="true" inert>
      <header class="annotation-panel-header">
        <div>
          <strong>笔记与标记</strong>
          <span class="annotation-book-title">请先打开一本书</span>
        </div>
        <button class="icon-button annotation-close" type="button" title="收起笔记工具" aria-label="收起笔记工具">${icons.close}</button>
      </header>
      <div class="annotation-selection">
        <span class="selection-status">在原文中拖选文字，再设置格式</span>
        <textarea class="selection-comment" rows="2" aria-label="给选中的原文写旁注" placeholder="给这段原文写旁注（可选）"></textarea>
      </div>
      <div class="annotation-style-row" aria-label="批注样式">
        <label class="annotation-color" title="颜色"><input type="color" value="#d8913d" aria-label="批注颜色" /></label>
        <select class="annotation-font" title="字体" aria-label="批注字体">
          <option value="inherit" selected>跟随原文</option>
          <option value="serif">宋体</option>
          <option value="sans-serif">黑体</option>
          <option value="cursive">手写体</option>
          <option value="monospace">等宽体</option>
        </select>
        <select class="annotation-size" title="字号" aria-label="批注字号">
          <option value="85">小</option>
          <option value="100" selected>标准</option>
          <option value="120">大</option>
          <option value="150">特大</option>
        </select>
      </div>
      <div class="annotation-format-row" aria-label="原文格式">
        <button type="button" class="format-toggle selected" data-format="highlight" title="荧光标记" aria-pressed="true">荧光</button>
        <button type="button" class="format-toggle" data-format="bold" title="加粗" aria-pressed="false">B</button>
        <button type="button" class="format-toggle" data-format="underline" title="下划线" aria-pressed="false"><u>U</u></button>
        <button type="button" class="format-toggle" data-format="wave" title="波浪线" aria-pressed="false">﹏</button>
        <button type="button" class="format-toggle" data-format="box" title="加框" aria-pressed="false">□</button>
        <button type="button" class="apply-text-mark">应用到选中文本</button>
      </div>
      <div class="annotation-tool-row" aria-label="自由笔记工具">
        <button type="button" class="annotation-tool selected" data-tool="read" title="阅读、拖选原文" aria-pressed="true">阅读</button>
        <button type="button" class="annotation-tool" data-tool="text" title="点击页面添加自由文字" aria-pressed="false">文字</button>
        <button type="button" class="annotation-tool" data-tool="pen" title="在页面上自由涂鸦" aria-pressed="false">画笔</button>
        <button type="button" class="annotation-tool" data-tool="eraser" title="点击自由文字或涂鸦删除" aria-pressed="false">橡皮</button>
        <label class="pen-width" title="画笔粗细"><span>粗细</span><input type="range" min="1" max="10" value="3" /></label>
        <button type="button" class="icon-button annotation-undo" title="撤销这本书最后一条笔记" aria-label="撤销最后一条笔记">${icons.undo}</button>
      </div>
      <div class="annotation-help">阅读模式可拖选原文；文字/画笔模式下点击页面；自由文字可直接编辑，拖动左上角手柄可移动。</div>
      <div class="annotation-list" aria-label="本书笔记清单"></div>
    </section>

    <button class="left-library-handle" type="button" aria-label="点击打开本地书库" aria-expanded="false" title="点击打开书库">
      <span>书库</span>${icons.forward}
    </button>
    <aside class="library-drawer" aria-label="本地书库" aria-hidden="true" inert>
      <div class="drawer-header">
        <div>
          <p class="eyebrow">舒适阅读书库</p>
          <h1>我的书库 <span class="book-count">0</span></h1>
        </div>
        <button class="icon-button drawer-close" type="button" title="收起书库" aria-label="收起书库">${icons.close}</button>
      </div>
      <label class="search-box">
        ${icons.search}
        <input type="search" placeholder="搜索书名或作者" autocomplete="off" />
      </label>
      <div class="library-actions" ${isDesktop ? "" : "hidden"}>
        ${isDesktop ? `<button class="primary-action add-books" type="button">${icons.plus}<span>添加 EPUB</span></button>
        <button class="secondary-action add-folder" type="button" title="加入一个本地书库文件夹">${icons.folder}</button>
        <button class="secondary-action refresh-library" type="button" title="刷新本地书库">${icons.refresh}</button>` : ""}
      </div>
      <div class="library-audit" role="status" aria-live="polite">
        <strong>正在核对全部书籍…</strong>
        <span>将递归扫描所有已登记书库</span>
      </div>
      <div class="library-list" role="list"></div>
      <div class="drawer-footer">
        <span class="status-dot"></span>
        <span>本地保存 · 自动记忆位置</span>
      </div>
    </aside>

    <section class="reader-grid panes-1" aria-label="多书阅读区"></section>
    <div class="toast" role="status"></div>
    <dialog class="jump-dialog" aria-labelledby="jump-dialog-title" aria-hidden="true" inert>
      <form class="jump-card">
        <div class="jump-card-header">
          <div>
            <strong id="jump-dialog-title">跳转到页</strong>
            <span class="jump-book-title">当前书籍</span>
          </div>
          <button class="icon-button jump-close" type="button" title="取消跳转" aria-label="取消跳转">${icons.close}</button>
        </div>
        <label class="jump-page-field">
          <span>页码</span>
          <input class="jump-page-input" type="number" min="1" step="1" inputmode="numeric" />
          <em>/ <span class="jump-page-total">—</span></em>
        </label>
        <div class="jump-card-actions">
          <button class="secondary-action jump-cancel" type="button">取消</button>
          <button class="primary-action jump-submit" type="submit">跳转</button>
        </div>
      </form>
    </dialog>
    <div class="boot-screen">
      <div class="boot-mark">${icons.book}</div>
      <strong>正在整理本地书库</strong>
      <span>书籍和阅读位置都只保存在这台电脑</span>
    </div>
  </div>
`;

const shell = requireElement<HTMLElement>(app, ".app-shell");

const drawer = requireElement<HTMLElement>(app, ".library-drawer");
const libraryHandle = requireElement<HTMLButtonElement>(app, ".left-library-handle");
const readerGrid = requireElement<HTMLElement>(app, ".reader-grid");
const libraryList = requireElement<HTMLElement>(app, ".library-list");
const libraryAudit = requireElement<HTMLElement>(app, ".library-audit");
const activeTitle = requireElement<HTMLElement>(app, ".active-book-title");
const fontScaleLabel = requireElement<HTMLElement>(app, ".font-scale-label");
const themeCycleButton = requireElement<HTMLButtonElement>(app, ".theme-cycle");
const toast = requireElement<HTMLElement>(app, ".toast");
const bootScreen = requireElement<HTMLElement>(app, ".boot-screen");
const searchInput = requireElement<HTMLInputElement>(app, ".search-box input");
const annotationPanel = requireElement<HTMLElement>(app, ".annotation-panel");
const annotationToggle = requireElement<HTMLButtonElement>(app, ".annotation-toggle");
const annotationBookTitle = requireElement<HTMLElement>(app, ".annotation-book-title");
const selectionStatus = requireElement<HTMLElement>(app, ".selection-status");
const selectionComment = requireElement<HTMLTextAreaElement>(app, ".selection-comment");
const annotationColorInput = requireElement<HTMLInputElement>(app, ".annotation-color input");
const annotationFontSelect = requireElement<HTMLSelectElement>(app, ".annotation-font");
const annotationSizeSelect = requireElement<HTMLSelectElement>(app, ".annotation-size");
const penWidthInput = requireElement<HTMLInputElement>(app, ".pen-width input");
const annotationList = requireElement<HTMLElement>(app, ".annotation-list");
const jumpDialog = requireElement<HTMLDialogElement>(app, ".jump-dialog");
const jumpForm = requireElement<HTMLFormElement>(app, ".jump-card");
const jumpInput = requireElement<HTMLInputElement>(app, ".jump-page-input");
const jumpTotal = requireElement<HTMLElement>(app, ".jump-page-total");
const jumpBookTitle = requireElement<HTMLElement>(app, ".jump-book-title");
let jumpReturnFocus: HTMLElement | null = null;

const runtimes: PaneRuntime[] = Array.from({ length: MAX_PANES }, () => ({
  readingMode: 'paged',
  streamed: false,
  opening: null,
  finishOpening: null,
  bookId: null,
  book: null,
  rendition: null,
  resizeObserver: null,
  resizeTimer: null,
  resizeFrame: null,
  resizeActive: false,
  resizeAnchorCfi: null,
  layoutAnchorCfi: null,
  preserveSemanticFocus: false,
  pendingStageWidth: 0,
  pendingStageHeight: 0,
  generation: 0,
  currentPage: 0,
  endPage: 0,
  totalPages: 0,
  percent: 0,
  cfi: null,
  pageMode: 0,
  actualPageCount: 1,
  effectiveFontScale: 100,
  layoutWidth: 0,
  layoutHeight: 0,
  viewportWidth: 0,
  viewportHeight: 0,
  viewportScale: 1,
  restoringLocation: false,
  paginationGeneration: 0,
  paginationTimer: null,
  atomicFitFrame: null,
  lazyPageCount: 0,
  lazyPageUrls: new Map<number, string>(),
  lazyPageLoading: new Map<number, Promise<string>>(),
  lazyPageGeneration: 0,
}));

const sourceDigests=new Map<string,string>();
const contentDigests=new Map<string,string>();
const sourceMismatches=new Set<string>();
function editionHeld(bookId:string|null):boolean{return Boolean(bookId&&sourceMismatches.has(bookId));}
function annotationEditionAvailable(bookId:string|null):boolean{
  if(!editionHeld(bookId))return true;
  showToast("书文件已换版。旧批注完整保留，待核对后才能修改或定位；可恢复原 EPUB 后重新打开。","error");return false;
}

let snapshot: AppSnapshot = {
  books: [],
  progress: {},
  session: {
    paneCount: 1,
    paneBookIds: [null, null, null, null],
    activePane: 0,
    theme: "paper",
    fontScale: 100,
    readerFont: "serif",
  },
  libraryRoots: [],
  storagePath: "",
  scan: {
    rootsScanned: 0,
    missingRoots: 0,
    filesSeen: 0,
    epubCandidates: 0,
    loadedCandidates: 0,
    booksLoaded: 0,
    added: 0,
    updated: 0,
    duplicates: 0,
    unreadable: 0,
    otherBookFiles: 0,
    ignoredTrees: 0,
    issues: [],
  },
};
let toastTimer: number | null = null;
let sessionSaveTimer: number | null = null;
const progressSaveTimers: Array<number | null> = Array.from({length:MAX_PANES},()=>null);
const progressWrites:Array<Promise<void>>=Array.from({length:MAX_PANES},()=>Promise.resolve());
const bookOpenSequence=Array.from({length:MAX_PANES},()=>0);
const pendingSelections: Array<PendingSelection | null> = Array.from({length:MAX_PANES},()=>null);
const activeFormats = new Set<keyof AnnotationTextStyle>(["highlight"]);
let annotationTool: AnnotationTool = "read";
let drawingDraft: DrawingDraft | null = null;
let noteDragState: NoteDragState | null = null;
let drawingFrame: number | null = null;

function escapeHtml(value: string): string {
  return value.replace(
    /[&<>"]/g,
    (character) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[
        character
      ] ?? character,
  );
}

function getBook(bookId: string | null): BookRecord | undefined {
  return snapshot.books.find((book) => book.id === bookId);
}

function normalizeSession(session: ReaderSession): ReaderSession {
  const paneBookIds = [...(session.paneBookIds ?? [])];
  while (paneBookIds.length < MAX_PANES) paneBookIds.push(null);
  return {
    workspace:session.workspace??null,
    paneCount: Math.min(MAX_PANES, Math.max(1, Number(session.paneCount) || 1)),
    paneBookIds: paneBookIds.slice(0, MAX_PANES),
    activePane: Math.min(
      Math.max(0, Number(session.activePane) || 0),
      Math.min(MAX_PANES, Math.max(1, Number(session.paneCount) || 1)) - 1,
    ),
    theme: ["paper", "light", "night", "contrast"].includes(session.theme)
      ? session.theme
      : "night",
    fontScale: Math.min(180, Math.max(70, Number(session.fontScale) || 100)),
    readerFont: ["serif", "sans", "publisher"].includes(session.readerFont) ? session.readerFont : "serif",
    lineHeight: session.lineHeight?Math.min(2.15,Math.max(1.35,Number(session.lineHeight))):null,
    contentWidth: [34,44,56].includes(Number(session.contentWidth))?Number(session.contentWidth):44,
  };
}

const ALLOWED_ANNOTATION_FONTS = new Set(["inherit", "serif", "sans-serif", "cursive", "monospace"]);

function safeAnnotationColor(value: unknown): string {
  return typeof value === "string" && /^#[0-9a-f]{6}$/i.test(value) ? value : "#d8913d";
}

function safeAnnotationFont(value: unknown): string {
  return typeof value === "string" && ALLOWED_ANNOTATION_FONTS.has(value) ? value : "inherit";
}

function clampNumber(value: unknown, minimum: number, maximum: number, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? Math.min(maximum, Math.max(minimum, parsed)) : fallback;
}

function textStyleFromRecord(record: Record<string, unknown>): AnnotationTextStyle {
  return {
    color: safeAnnotationColor(record.color),
    fontFamily: safeAnnotationFont(record.fontFamily),
    fontSize: clampNumber(record.fontSize, 70, 220, 100),
    highlight: Boolean(record.highlight),
    bold: Boolean(record.bold),
    underline: Boolean(record.underline),
    wave: Boolean(record.wave),
    box: Boolean(record.box),
  };
}

function normalizeAnnotations(raw: unknown): ReaderAnnotation[] {
  if (!Array.isArray(raw)) return [];
  const normalized: ReaderAnnotation[] = [];
  for (const entry of raw) {
    if (!entry || typeof entry !== "object") continue;
    const record = entry as Record<string, unknown>;
    const id = typeof record.id === "string" && record.id.length <= 120
      ? record.id
      : crypto.randomUUID();
    const createdAt = clampNumber(record.createdAt, 0, Number.MAX_SAFE_INTEGER, Date.now());
    if (record.kind === "text-mark" && typeof record.cfiRange === "string") {
      normalized.push({
        ...record,
        ...textStyleFromRecord(record),
        id,
        kind: "text-mark",
        cfiRange: record.cfiRange,
        selectedText: typeof record.selectedText === "string" ? record.selectedText.slice(0, 12000) : "",
        comment: typeof record.comment === "string" ? record.comment.slice(0, 12000) : "",
        createdAt,
      });
      continue;
    }
    if (record.kind === "free-text") {
      normalized.push({
        ...record,
        ...textStyleFromRecord(record),
        id,
        kind: "free-text",
        anchorCfi: typeof record.anchorCfi === "string" ? record.anchorCfi : null,
        anchorVersion: record.anchorVersion === 2 ? 2 : undefined,
        progression: clampNumber(record.progression, 0, 1, 0),
        x: clampNumber(record.x, -1, 2, 0.1),
        y: clampNumber(record.y, 0, 1, 0.12),
        text: typeof record.text === "string" ? record.text.slice(0, 24000) : "",
        createdAt,
      });
      continue;
    }
    if (record.kind === "drawing" && Array.isArray(record.points)) {
      const points = record.points
        .slice(0, 8000)
        .flatMap((point): DrawingPoint[] => {
          if (!point || typeof point !== "object") return [];
          const item = point as Record<string, unknown>;
          return [{
            x: clampNumber(item.x, -2, 3, 0),
            y: clampNumber(item.y, -1, 2, 0),
          }];
        });
      if (points.length < 2) continue;
      normalized.push({
        ...record,
        id,
        kind: "drawing",
        anchorCfi: typeof record.anchorCfi === "string" ? record.anchorCfi : null,
        anchorVersion: record.anchorVersion === 2 ? 2 : undefined,
        progression: clampNumber(record.progression, 0, 1, 0),
        color: safeAnnotationColor(record.color),
        strokeWidth: clampNumber(record.strokeWidth, 1, 20, 3),
        points,
        createdAt,
      });
    }
  }
  return normalized;
}

function normalizeProgress(progress: Record<string, BookProgress>): Record<string, BookProgress> {
  for (const entry of Object.values(progress)) {
    entry.pageMode = Math.min(10, Math.max(0, Number(entry.pageMode) || 0));
    entry.annotations = normalizeAnnotations(entry.annotations);
  }
  return progress;
}

function annotationsForBook(bookId: string | null): ReaderAnnotation[] {
  if (!bookId) return [];
  const progress = snapshot.progress[bookId];
  if (!progress) return [];
  if (!Array.isArray(progress.annotations)) progress.annotations = [];
  return progress.annotations;
}

function currentAnnotationStyle(): AnnotationTextStyle {
  return {
    color: safeAnnotationColor(annotationColorInput.value),
    fontFamily: safeAnnotationFont(annotationFontSelect.value),
    fontSize: clampNumber(annotationSizeSelect.value, 70, 220, 100),
    highlight: activeFormats.has("highlight"),
    bold: activeFormats.has("bold"),
    underline: activeFormats.has("underline"),
    wave: activeFormats.has("wave"),
    box: activeFormats.has("box"),
  };
}

function pageModeForBook(bookId: string | null): number {
  if (!bookId) return 0;
  return Math.min(10, Math.max(0, Number(snapshot.progress[bookId]?.pageMode) || 0));
}

function readingPreferences(index:number){
  const id=runtimes[index]?.bookId??snapshot.session.paneBookIds[index],saved=id?snapshot.progress[id]?.readingPreferences:null;
  return {
    fontScale:Math.max(70,Math.min(180,saved?.fontScale??snapshot.session.fontScale)),
    readerFont:saved?.readerFont&&['serif','sans','publisher'].includes(saved.readerFont)?saved.readerFont:snapshot.session.readerFont,
    lineHeight:clampNumber(saved?.lineHeight??snapshot.session.lineHeight??1.72,1.35,2.15,1.72),
    contentWidth:[34,44,56].includes(Number(saved?.contentWidth))?Number(saved?.contentWidth):snapshot.session.contentWidth??44,
  };
}
function changeReadingPreference<K extends keyof NonNullable<BookProgress['readingPreferences']>>(key:K,value:NonNullable<BookProgress['readingPreferences']>[K]):void{
  const index=snapshot.session.activePane,id=runtimes[index].bookId;if(!id)return;
  const progress=ensureBookProgress(id,index);progress.readingPreferences={...readingPreferences(index),[key]:value};
  const rendition=runtimes[index].rendition;if(rendition)for(const contents of visibleContents(rendition))applyReadingTheme(contents);
  void reflowPane(index,true);syncReadingSettings();savePaneProgress(index,0);
}
function syncReadingSettings():void{
  const index=snapshot.session.activePane,prefs=readingPreferences(index),runtime=runtimes[index];
  fontScaleLabel.textContent=`${prefs.fontScale}%`;
  requireElement<HTMLSelectElement>(app,'.reader-font').value=prefs.readerFont;
  requireElement<HTMLSelectElement>(app,'.reading-mode').value=runtime.readingMode;
  requireElement<HTMLInputElement>(app,'.reader-line-height').value=String(prefs.lineHeight);
  requireElement<HTMLSelectElement>(app,'.reader-content-width').value=String(prefs.contentWidth);
  const pages=requireElement<HTMLSelectElement>(app,'.settings-page-mode');pages.value=String(runtime.pageMode);pages.disabled=runtime.readingMode==='scroll'||!runtime.bookId;
  requireElement<HTMLElement>(app,'.settings-book-title').textContent=getBook(runtime.bookId)?.title??'先从书库打开一本书';
}
function syncReadingChrome():void{
  for(let index=0;index<MAX_PANES;index++){
    const footer=paneElement(index).querySelector<HTMLElement>('.pane-footer')!;
    const visible=toolsVisible&&index===snapshot.session.activePane;
    footer.inert=!visible;footer.setAttribute('aria-hidden',String(!visible));
  }
}
function setToolsVisible(visible:boolean):void{
  toolsVisible=visible;shell.classList.toggle('tools-visible',visible);
  const toolbar=requireElement<HTMLElement>(app,'.top-toolbar');toolbar.classList.toggle('visible',visible);toolbar.inert=!visible;toolbar.setAttribute('aria-hidden',String(!visible));
  const trigger=requireElement<HTMLButtonElement>(app,'.tools-toggle');trigger.setAttribute('aria-expanded',String(visible));trigger.setAttribute('aria-label',visible?'收起阅读工具':'打开阅读工具');
  if(!visible){closeBookNavigation();setReadingSettings(false);if(toolbar.contains(document.activeElement))trigger.focus({preventScroll:true});}
  syncReadingChrome();
}

function showToast(message: string, kind: "normal" | "error" = "normal"): void {
  toast.textContent = message;
  toast.dataset.kind = kind;
  toast.classList.add("visible");
  if (toastTimer !== null) window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => toast.classList.remove("visible"), 2600);
}



function openDrawer(): void {
  setToolsVisible(false);
  setAnnotationPanelVisible(false);
  workspace?.showPanel(false,false);
  closeBookNavigation();
  setReadingSettings(false);
  drawer.inert = false;
  drawer.setAttribute("aria-hidden", "false");
  app.querySelector(".library-toggle")?.setAttribute("aria-expanded", "true");
  drawer.classList.add("visible");
  shell.classList.add("drawer-open");
  libraryHandle.setAttribute("aria-expanded", "true");
  searchInput.focus({preventScroll:true});
}

function closeDrawer(): void {
  const returnFocus=drawer.contains(document.activeElement);
  drawer.inert = true;
  drawer.setAttribute("aria-hidden", "true");
  app.querySelector(".library-toggle")?.setAttribute("aria-expanded", "false");
  drawer.classList.remove("visible");
  shell.classList.remove("drawer-open");
  libraryHandle.setAttribute("aria-expanded", "false");
  if(returnFocus)app.querySelector<HTMLButtonElement>('.workspace-library')?.focus({preventScroll:true});
}

function closeJumpDialog(): void {
  if (!jumpDialog.open) return;
  jumpDialog.close();
  jumpDialog.classList.remove("visible");
  jumpDialog.setAttribute("aria-hidden", "true");
  jumpDialog.inert = true;
  const trigger = jumpReturnFocus;
  jumpReturnFocus = null;
  if (trigger?.isConnected && !trigger.closest("[inert]")) trigger.focus({ preventScroll: true });
}

function openJumpDialog(index = snapshot.session.activePane): void {
  if (jumpDialog.open) return;
  const runtime = runtimes[index];
  const book = getBook(runtime.bookId);
  if (!book || (!runtime.rendition && runtime.lazyPageCount <= 0)) {
    showToast("请先打开一本书", "error");
    return;
  }
  const starter = document.activeElement;
  jumpReturnFocus = starter instanceof HTMLElement && starter !== document.body
    ? starter : paneElement(index).querySelector<HTMLElement>(".page-jump-button");
  setActivePane(index);
  jumpBookTitle.textContent = book.title;
  const stableLocations = runtime.readingMode==='scroll' || runtime.book && linearSectionCount(runtime.book) > 1;
  requireElement<HTMLElement>(jumpDialog, ".jump-card-header strong").textContent = stableLocations ? "跳转到阅读位置" : "跳转到页";
  requireElement<HTMLElement>(jumpDialog, ".jump-page-field > span").textContent = stableLocations ? "位置" : "页码";
  jumpTotal.textContent = runtime.totalPages > 0 ? String(runtime.totalPages) : "正在生成";
  jumpInput.max = runtime.totalPages > 0 ? String(runtime.totalPages) : "";
  jumpInput.value = runtime.currentPage > 0 ? String(runtime.currentPage) : "1";
  const bodyPosition = runtime.bookId ? learningReadingProgress(runtime.bookId, runtime.book, runtime.cfi,index) : null;
  if (bodyPosition) {
    requireElement<HTMLElement>(jumpDialog, ".jump-card-header strong").textContent = "跳转到正文位置";
    jumpTotal.textContent = String(bodyPosition.total);
    jumpInput.max = String(bodyPosition.total);
    jumpInput.value = String(bodyPosition.position);
  }
  jumpDialog.inert = false;
  jumpDialog.setAttribute("aria-hidden", "false");
  jumpDialog.showModal();
  jumpDialog.classList.add("visible");
  window.requestAnimationFrame(() => {
    if (!jumpDialog.open) return;
    jumpInput.focus();
    jumpInput.select();
  });
}

function buildPaneShells(): void {
  readerGrid.innerHTML = Array.from({ length: MAX_PANES }, (_, index) => `
    <article class="reader-pane" data-pane-index="${index}">
      <header class="pane-header">
        <button class="pane-drag-handle" type="button" title="拖动摆放这本书；也可在阅读现场中用按钮调整" aria-label="拖动摆放这本书">⠿</button>
        <div class="pane-book-meta">
          <strong>空白阅读窗格</strong>
          <span>从左侧书库选择一本书</span>
        </div>
        <label class="pane-reading-mode"><span class="sr-only">阅读方式</span><select aria-label="这本书的阅读方式"><option value="paged">翻页</option><option value="scroll">连续滚动</option></select></label>
        <label class="pane-page-mode" title="这本书同屏显示的页数">
          <span>同屏</span>
          <select aria-label="这本书的同屏页数">
            <option value="0">自动</option>
            ${Array.from({ length: 10 }, (_, pageIndex) => `<option value="${pageIndex + 1}">${pageIndex + 1} 页</option>`).join("")}
          </select>
        </label>
        <button class="icon-button pane-focus" type="button" title="临时专注 / 返回对照" aria-label="临时专注这本书">${icons.fullscreen}</button>
        <button class="icon-button pane-close" type="button" title="关闭这本书" aria-label="关闭这本书">${icons.close}</button>
      </header>
      <div class="pane-stage">
        <div class="empty-pane">
          <div class="empty-mark">${icons.book}</div>
          <strong>${isDesktop ? "把一本书放到这里" : "从书目选一本书"}</strong>
          <span>${isDesktop ? "点击工具栏中的书库按钮" : "读过的书会留在我的书籍中"}</span>
          <button type="button" class="empty-library-button">${isDesktop ? "浏览本地书库" : "发现书籍"}</button>
        </div>
        <div class="pane-loading"><span></span><em>正在排版…</em></div>
        <div class="pane-load-failure" role="status"><strong>这次未能打开内容</strong><p>原书籍、位置和个人记录仍在。检查连接或文件后可以重试。</p><button class="retry-book" type="button">重试打开</button><button class="empty-library-button" type="button">回到书库</button></div>
        <div class="epub-host" id="epub-host-${index}"></div>
        <div class="annotation-layer" data-pane-index="${index}" aria-label="自由笔记与涂鸦画布">
          <svg class="drawing-layer" aria-hidden="true"></svg>
          <div class="free-note-layer"></div>
        </div>
        <button class="page-hotspot page-hotspot-prev" type="button" aria-label="上一页">${icons.back}</button>
        <button class="page-hotspot page-hotspot-group-prev" type="button" title="连翻 N 页（按这本书当前的同屏页数）" aria-label="连翻 N 页，按这本书当前的同屏页数后退一组">${icons.groupBack}<span class="hotspot-count">1</span></button>
        <button class="page-hotspot page-hotspot-next" type="button" aria-label="下一页">${icons.forward}</button>
        <button class="page-hotspot page-hotspot-group-next" type="button" title="连翻 N 页（按这本书当前的同屏页数）" aria-label="连翻 N 页，按这本书当前的同屏页数前进一组">${icons.groupForward}<span class="hotspot-count">1</span></button>
      </div>
      <footer class="pane-footer">
        <button class="page-label page-jump-button" type="button" title="点击跳转（G）">尚未打开</button>
        <div class="pane-chapter-boundary" hidden>
          <button type="button" class="chapter-boundary-prev" hidden>← 上一章</button>
          <button type="button" class="chapter-boundary-next" hidden>下一章 →</button>
        </div>
        <div class="progress-track"><span></span></div>
        <div class="percent-label">0%</div>
      </footer>
    </article>
  `).join("");
}

function paneElement(index: number): HTMLElement {
  const element = readerGrid.querySelector<HTMLElement>(
    `.reader-pane[data-pane-index="${index}"]`,
  );
  if (!element) throw new Error(`阅读窗格 ${index + 1} 不存在`);
  return element;
}

function destroyRuntime(index: number, preserveBookId = true, savePosition=true): void {
  const runtime = runtimes[index];
  if(progressSaveTimers[index]!==null){clearTimeout(progressSaveTimers[index]!);progressSaveTimers[index]=null;}
  const last=savePosition?makePaneProgress(index):null;if(last)void writePaneProgress(index,last).catch(error=>showToast(String(error),'error'));
  runtime.finishOpening?.();runtime.opening=null;runtime.finishOpening=null;
  runtime.generation += 1;
  runtime.resizeObserver?.disconnect();
  runtime.resizeObserver = null;
  if (runtime.resizeTimer !== null) window.clearTimeout(runtime.resizeTimer);
  runtime.resizeTimer = null;
  if (runtime.resizeFrame !== null) window.cancelAnimationFrame(runtime.resizeFrame);
  runtime.resizeFrame = null;
  runtime.resizeActive = false;
  runtime.resizeAnchorCfi = null;
  runtime.layoutAnchorCfi = null;
  runtime.preserveSemanticFocus = false;
  runtime.streamed = false;
  runtime.pendingStageWidth = 0;
  runtime.pendingStageHeight = 0;
  runtime.paginationGeneration += 1;
  if (runtime.paginationTimer !== null) window.clearTimeout(runtime.paginationTimer);
  runtime.paginationTimer = null;
  if (runtime.atomicFitFrame !== null) window.cancelAnimationFrame(runtime.atomicFitFrame);
  runtime.atomicFitFrame = null;
  runtime.rendition?.destroy();
  runtime.book?.destroy();
  runtime.rendition = null;
  runtime.book = null;
  runtime.currentPage = 0;
  runtime.endPage = 0;
  runtime.totalPages = 0;
  runtime.lazyPageCount = 0;
  runtime.percent = 0;
  runtime.cfi = null;
  runtime.pageMode = 0;
  runtime.readingMode = 'paged';
  runtime.actualPageCount = 1;
  runtime.effectiveFontScale = 100;
  runtime.layoutWidth = 0;
  runtime.layoutHeight = 0;
  runtime.viewportWidth = 0;
  runtime.viewportHeight = 0;
  runtime.viewportScale = 1;
  runtime.restoringLocation = false;
  pendingSelections[index] = null;
  if (!preserveBookId) runtime.bookId = null;
  const host = paneElement(index).querySelector<HTMLElement>(".epub-host");
  if (host) host.replaceChildren();
  const overlay = annotationOverlay(index);
  overlay?.querySelector(".drawing-layer")?.replaceChildren();
  overlay?.querySelector(".free-note-layer")?.replaceChildren();
}

function updatePaneHeader(index: number): void {
  const pane = paneElement(index);
  const runtime = runtimes[index];
  const book = getBook(runtime.bookId);
  const empty=pane.querySelector<HTMLElement>('.empty-pane');if(empty){empty.inert=Boolean(book);empty.setAttribute('aria-hidden',String(Boolean(book)));}
  const title = pane.querySelector<HTMLElement>(".pane-book-meta strong");
  const subtitle = pane.querySelector<HTMLElement>(".pane-book-meta span");
  const pageSelect = pane.querySelector<HTMLSelectElement>(".pane-page-mode select");
  pane.dataset.readingMode = runtime.readingMode;
  pane.dataset.pageMode=String(runtime.pageMode);
  const modeSelect=pane.querySelector<HTMLSelectElement>('.pane-reading-mode select');
  if(modeSelect){modeSelect.value=runtime.readingMode;modeSelect.disabled=!book||Boolean(book.lazyPages);}
  const groupPrev = pane.querySelector<HTMLButtonElement>(".page-hotspot-group-prev");
  const groupNext = pane.querySelector<HTMLButtonElement>(".page-hotspot-group-next");
  if (title) title.textContent = book?.title ?? "空白阅读窗格";
  if (subtitle)
    subtitle.textContent = book
      ? editionHeld(book.id) ? "版本已变化 · 原位置与批注待核对" : `${book.author} · ${book.catalogSource ? /^https?:\/\//.test(book.path)?"按章取得内容":"本机正文 · 资料按需取得" : book.lazyPages ? "按需加载 EPUB" : "本机书籍"}`
      : "从左侧书库选择一本书";
  if (pageSelect) {
    const mode = pageModeForBook(book?.id ?? null);
    runtime.pageMode = mode;
    pageSelect.value = String(mode);
    pageSelect.disabled = !book || runtime.readingMode === 'scroll';
    const autoOption = pageSelect.querySelector<HTMLOptionElement>('option[value="0"]');
    if (autoOption) autoOption.textContent = `自动 · ${runtime.actualPageCount} 页`;
    pageSelect.title = mode === 0
      ? `当前由窗口宽度自动显示 ${runtime.actualPageCount} 页，排版字号 ${Math.round(runtime.effectiveFontScale)}%`
      : `固定同屏显示 ${mode} 页；排版字号 ${Math.round(runtime.effectiveFontScale)}%，窗口画布等比缩放 ${Math.round(runtime.viewportScale * 100)}%，拖动窗口不重新分页`;
  }
  const groupSize = Math.max(1, runtime.actualPageCount);
  if (groupPrev) {
    groupPrev.title = `连翻 ${groupSize} 页`;
    groupPrev.setAttribute("aria-label", `连翻 ${groupSize} 页，后退一个完整页组`);
    const count = groupPrev.querySelector<HTMLElement>(".hotspot-count");
    if (count) count.textContent = String(groupSize);
  }
  if (groupNext) {
    groupNext.title = `连翻 ${groupSize} 页`;
    groupNext.setAttribute("aria-label", `连翻 ${groupSize} 页，前进一个完整页组`);
    const count = groupNext.querySelector<HTMLElement>(".hotspot-count");
    if (count) count.textContent = String(groupSize);
  }
  pane.classList.toggle("has-book", Boolean(book));
  pane.classList.toggle("active", index === snapshot.session.activePane);
  pane.setAttribute('aria-label',book?.title??'阅读区域');
}

function updatePaneProgress(index: number): void {
  const pane = paneElement(index);
  const runtime = runtimes[index];
  const pageLabel = pane.querySelector<HTMLElement>(".page-label");
  const percentLabel = pane.querySelector<HTMLElement>(".percent-label");
  const progressBar = pane.querySelector<HTMLElement>(".progress-track span");
  const percent = Math.min(1, Math.max(0, runtime.percent || 0));
  if (pageLabel) {
    pageLabel.textContent = runtime.bookId
      ? runtime.totalPages > 0
        ? runtime.endPage > runtime.currentPage
          ? `第 ${Math.max(1, runtime.currentPage)}–${runtime.endPage} / ${runtime.totalPages} 页`
          : `第 ${Math.max(1, runtime.currentPage)} / ${runtime.totalPages} 页`
        : "正在生成页码…"
      : "尚未打开";
    if (runtime.book && (runtime.readingMode==='scroll'||linearSectionCount(runtime.book) > 1)) {
      pageLabel.textContent = runtime.book.locations.length() > 0
        ? `阅读位置 ${Math.max(1, runtime.currentPage)} / ${runtime.totalPages}`
        : "正在建立全书位置索引…";
    }
  }
  if (percentLabel) percentLabel.textContent = `${(percent * 100).toFixed(1)}%`;
  if (progressBar) progressBar.style.width = `${percent * 100}%`;
  const studyPosition = runtime.bookId ? learningReadingProgress(runtime.bookId, runtime.book, runtime.cfi,index) : null;
  if (studyPosition) {
    if (pageLabel) pageLabel.textContent = studyPosition.text;
    if (percentLabel) percentLabel.textContent = studyPosition.percent === null ? studyPosition.kind==='reference'?"资料阅读":"已保存位置" : `${(studyPosition.percent * 100).toFixed(1)}%`;
    if (progressBar) progressBar.style.width = studyPosition.percent === null ? "0%" : `${studyPosition.percent * 100}%`;
  }
}

function updateActiveUi(): void {
  refreshLearningButton();
  const activeBookId = snapshot.session.paneBookIds[snapshot.session.activePane] ?? null;
  activeTitle.textContent = getBook(activeBookId)?.title ?? "尚未打开书籍";
  activeTitle.title = getBook(activeBookId)?.path ?? "当前选中的阅读窗格";
  readerGrid
    .querySelectorAll<HTMLElement>(".reader-pane")
    .forEach((pane, index) => pane.classList.toggle("active", index === snapshot.session.activePane));
  if (annotationPanel.classList.contains("visible")) renderAnnotationPanel();
  workspace?.setActive(snapshot.session.activePane);
  syncReadingSettings();
  syncReadingChrome();
}

function updateLayoutUi(): void {
  const ids=snapshot.session.paneBookIds.flatMap((id,i)=>id?[i]:[]);
  snapshot.session.paneCount=ids.length?Math.max(...ids)+1:1;
  workspace?.sync(snapshot.session.workspace,ids,snapshot.session.activePane);
  if(workspace)snapshot.session.workspace=workspace.state;
  readerGrid.dataset.panes = String(ids.length);
  for (let index = 0; index < MAX_PANES; index += 1) updatePaneHeader(index);
  app.querySelectorAll<HTMLButtonElement>("[data-pane-count]").forEach((button) => {
    const selected = Number(button.dataset.paneCount) === snapshot.session.paneCount;
    button.classList.toggle("selected", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
  updateActiveUi();
}

function sessionPayload(): ReaderSession {
  return {
    ...snapshot.session,
    paneBookIds: snapshot.session.paneBookIds.slice(0, MAX_PANES),
  };
}

function persistSession(): void {
  if (sessionSaveTimer !== null) window.clearTimeout(sessionSaveTimer);
  sessionSaveTimer = window.setTimeout(() => {
    void invoke("save_session", { session: sessionPayload() }).catch((error) =>
      showToast(String(error), "error"),
    );
  }, 180);
}

function makePaneProgress(index:number):{bookId:string;progress:BookProgress}|null{
  const runtime = runtimes[index];
  if(!runtime.bookId||editionHeld(runtime.bookId)||runtime.restoringLocation||(!runtime.cfi&&!runtime.lazyPageCount))return null;
    const existing = snapshot.progress[runtime.bookId];
    const bodyMetric = learningReadingProgress(runtime.bookId, runtime.book, runtime.cfi,index);
    const readingMetric: import('./reading-progress').ReadingMetric | null = bodyMetric
      ? {kind: bodyMetric.kind, position: bodyMetric.position, total: bodyMetric.total, percent: bodyMetric.percent}
      : runtime.book && linearSectionCount(runtime.book) > 1 && runtime.book.locations.length() === 0
        ? existing?.readingMetric ?? null
        : runtime.totalPages > 0 ? {
          kind: runtime.lazyPageCount > 0 || runtime.readingMode==='paged' && (runtime.book?.spine as InternalSpine)?.spineItems?.length === 1 ? 'page' : 'position',
          position: runtime.currentPage, total: runtime.totalPages, percent: runtime.percent,
        } : existing?.readingMetric ?? null;
    const progress: BookProgress = {
      ...existing,
      readingMode: runtime.readingMode,
      readingMetric,
      sourceSha256:sourceDigests.get(runtime.bookId)??existing?.sourceSha256??null,
      cfi: runtime.lazyPageCount > 0 ? null : runtime.cfi ?? existing?.cfi ?? null,
      page: runtime.currentPage || existing?.page || 0,
      totalPages: runtime.totalPages || existing?.totalPages || 0,
      percent: runtime.lazyPageCount > 0
        ? runtime.percent
        : runtime.cfi
          ? runtime.percent
          : existing?.percent ?? 0,
      updatedAt: Math.floor(Date.now() / 1000),
      pageMode: runtime.pageMode,
      annotations: existing?.annotations ?? [],
    };
    return {bookId:runtime.bookId,progress:structuredClone(progress)};
}
function writePaneProgress(index:number,value:{bookId:string;progress:BookProgress}):Promise<void>{
    snapshot.progress[value.bookId] = value.progress;
    renderLibrary(searchInput.value);
    const write=progressWrites[index].catch(()=>{}).then(()=>invoke<void>('save_progress',value));progressWrites[index]=write;return write;
}
function savePaneProgress(index:number,delay=240):void{
  if(progressSaveTimers[index]!==null)clearTimeout(progressSaveTimers[index]!);
  const generation=runtimes[index].generation;
  progressSaveTimers[index]=window.setTimeout(()=>{progressSaveTimers[index]=null;if(generation!==runtimes[index].generation)return;if(runtimes[index].restoringLocation||runtimes[index].resizeActive){savePaneProgress(index,240);return;}const value=makePaneProgress(index);if(value)void writePaneProgress(index,value).catch(error=>showToast(String(error),'error'));},delay);
}

function themeRules(theme: ThemeName,index=snapshot.session.activePane): Record<string, Record<string, string>> {
  const preferences=readingPreferences(index);
  const palette = {
    paper: {
      background: "#f8f3e9",
      text: "#2d2923",
      muted: "#655c50",
      accent: "#a85d3b",
      surface: "#f0e8dc",
      border: "#cfc2b0",
      rule: "#d7cbbb",
    },
    light: {
      background: "#ffffff",
      text: "#20242a",
      muted: "#66707b",
      accent: "#476e91",
      surface: "#f3f6f8",
      border: "#cbd3da",
      rule: "#d8dee4",
    },
    night: {
      background: "#1c1f23",
      text: "#d9dadd",
      muted: "#b2b7bd",
      accent: "#a3b5bd",
      surface: "#25292e",
      border: "#43484e",
      rule: "#33383e",
    },
    contrast: { background:"#08090b", text:"#ffffff", muted:"#e0e0e0", accent:"#ffe29a", surface:"#16191f", border:"#929292", rule:"#aaaaaa" },
  }[theme];
  const rules: Record<string, Record<string, string>> = {
    html: {
      "background-color": `${palette.background} !important`,
      color: `${palette.text} !important`,
      "color-scheme": `${(theme === "night" || theme === "contrast") ? "dark" : "light"} !important`,
    },
    body: {
      "background-color": `${palette.background} !important`,
      color: `${palette.text} !important`,
      "font-family": `${readingFontFamily(index)} !important`,
      "--reader-font-body": readingFontFamily(index),
      "line-height": `${preferences.lineHeight} !important`,
      "padding-left": "1em !important",
      "padding-right": "1em !important",
      "box-sizing": "border-box !important",
      "column-rule": `1px solid ${palette.rule} !important`,
    },
    "h1, h2, h3, h4, h5, h6": {
      color: `${palette.text} !important`,
      "line-height": "1.3 !important",
      "break-after": "avoid-column !important",
      "page-break-after": "avoid !important",
    },
    blockquote: {
      color: `${palette.muted} !important`,
      "background-color": `${palette.surface} !important`,
      "border-left-color": `${palette.accent} !important`,
    },
    figcaption: { color: `${palette.muted} !important` },
    "a, a:visited": { color: "inherit !important", "text-decoration-color": `${palette.accent} !important` },
    "pre, code": {
      "white-space": "pre-wrap !important",
      "overflow-wrap": "anywhere !important",
      color: `${palette.text} !important`,
      "background-color": `${palette.surface} !important`,
      "border-color": `${palette.border} !important`,
    },
    "menclose.comfortable-math-box, menclose[notation~='box']": {
      display: "inline-block !important",
      border: ".075em solid currentColor !important",
      padding: ".12em .3em !important",
      "box-sizing": "border-box !important",
    },
    "img, svg, video": {
      "max-width": "100% !important",
      height: "auto !important",
      "break-inside": "avoid-column !important",
      "page-break-inside": "avoid !important",
    },
    "figure, table, pre, .math-block, .math-display, .MathJax_Display, .katex-display, math[display='block']": {
      "break-inside": "avoid-column !important",
      "page-break-inside": "avoid !important",
    },
    "table, tr, thead, tbody": {
      "break-inside": "avoid-column !important",
      "page-break-inside": "avoid !important",
    },
    table: {
      "max-width": "100% !important",
      "border-collapse": "collapse !important",
      "background-color": "transparent !important",
      "border-color": `${palette.border} !important`,
    },
    "th, td": {
      color: `${palette.text} !important`,
      "border-color": `${palette.border} !important`,
    },
    th: {
      "background-color": `${palette.surface} !important`,
    },
    hr: {
      color: `${palette.rule} !important`,
      "border-color": `${palette.rule} !important`,
    },
    ".reader-annotation": {
      display: "inline !important",
      margin: "0 !important",
      padding: "0 !important",
      "box-decoration-break": "clone !important",
      "-webkit-box-decoration-break": "clone !important",
    },
  };
  if (theme === "night" || theme === "contrast") {
    rules["main, article, section, header, footer, nav, aside, div:not(.reader-annotation)"] = {
      color: `${palette.text} !important`,
      "background-color": "transparent !important",
      "border-color": `${palette.border} !important`,
    };
    rules["p, li, dt, dd, span:not(.reader-annotation)"] = {
      color: `${palette.text} !important`,
      "border-color": `${palette.border} !important`,
    };
    rules["mark:not(.reader-annotation)"] = {
      color: `${palette.text} !important`,
      "background-color": "rgba(224, 160, 119, 0.2) !important",
    };
  }

  if (preferences.readerFont !== "publisher") {
    rules["p, li, blockquote, h1, h2, h3, h4, h5, h6, td, th, figcaption, .algorithm"] = {
      ...(rules["p, li, blockquote, h1, h2, h3, h4, h5, h6, td, th, figcaption, .algorithm"] ?? {}),
      "font-family": `${readingFontFamily(index)} !important`,
    };
  } else { delete rules.body["font-family"]; delete rules.body["--reader-font-body"]; }
  rules["body:lang(en)"] = { "line-height": `${preferences.lineHeight} !important`, "hyphens": "auto" };
  rules["p, li"] = { "orphans": "2", "widows": "2", "word-break": "normal", "overflow-wrap": "break-word" };
  rules["p:lang(en), li:lang(en)"] = { "text-align": "start !important" };
  rules["h1, h2, h3, h4, h5, h6"] = { ...rules["h1, h2, h3, h4, h5, h6"], "border-color": `${palette.border} !important`, "overflow-wrap": "anywhere !important" };
  rules["h1 a, h2 a, h3 a, h4 a, h5 a, h6 a"] = { "text-decoration": "none !important" };
  rules["ul, ol"] = { "margin": ".5em 0 !important", "padding-inline-start": "1.25em !important" };
  rules[".math-inline"] = { "display": "inline !important", "vertical-align": "baseline !important" };
  rules[".math-inline math"] = { "display": "inline math !important" };
  rules[".reader-algorithm, .algorithm-cost"] = { "table-layout": "auto !important" };
  rules["math, math *"] = { "font-family": '"Cambria Math", "STIX Two Math", math !important' };
  rules["pre, code, kbd, samp"] = { ...rules["pre, code"], "font-family": '"Cascadia Code", Consolas, "Noto Sans SC", monospace !important' };
  rules[".algorithm, .reader-algorithm"] = { "background-color": `${palette.surface} !important`, "border-color": `${palette.border} !important`, "line-height": "1.4 !important", "break-inside": "auto !important", "page-break-inside": "auto !important" };
  rules[".algorithm tbody, .reader-algorithm tbody"] = { "break-inside": "auto !important", "page-break-inside": "auto !important" };
  rules[".algorithm td, .reader-algorithm td"] = { "font-family": 'Cambria, "Noto Serif SC", serif !important', "line-height": "1.4 !important", "vertical-align": "baseline !important", "border": "none !important", "padding-top": ".12em !important", "padding-bottom": ".12em !important", "padding-right": ".3em !important", "overflow-wrap": "break-word !important" };
  rules[".algorithm td:first-child, .reader-algorithm td:first-child"] = { "width": "2em !important", "min-width": "2em !important", "text-align": "right !important", "padding-left": ".2em !important", "padding-right": ".6em !important", "color": `${palette.muted} !important`, "font-size": ".85em !important" };
  rules[".algorithm td:last-child, .reader-algorithm td:last-child"] = { "width": "auto !important" };
  rules[".reader-algorithm .algorithm-title td"] = { "padding": ".55em .3em .4em !important", "text-align": "left !important", "font-weight": "600 !important", "font-size": "1em !important", "color": `${palette.text} !important` };
  rules[".reader-algorithm .algorithm-comment, .reader-algorithm .algorithm-comment *"] = { "color": `${palette.muted} !important` };
  rules[".algorithm-cost td:first-child, .algorithm-cost th:first-child"] = { "width": "2em !important", "text-align": "right !important" };
  rules[".algorithm-cost td:nth-child(3), .algorithm-cost th:nth-child(3)"] = { "width": "3em !important", "text-align": "center !important" };
  rules[".algorithm-cost td:nth-child(4), .algorithm-cost th:nth-child(4)"] = { "width": "5em !important", "text-align": "center !important" };
  rules[".algorithm-cost, .algorithm-cost tbody"] = { "break-inside": "auto !important", "page-break-inside": "auto !important" };
  rules[".algorithm-cost td, .algorithm-cost th"] = { "padding": ".2em .3em !important", "line-height": "1.4 !important" };
  rules["table.reader-flow-table, table.reader-flow-table tbody, table.reader-flow-table tfoot"] = { "break-inside": "auto !important", "page-break-inside": "auto !important" };
  rules["table.reader-flow-table tr"] = { "break-inside": "avoid-column !important", "page-break-inside": "avoid !important" };
  rules["table.reader-flow-table thead"] = { "display": "table-header-group !important", "break-inside": "avoid-column !important", "page-break-inside": "avoid !important" };
  rules[".algorithm-title"] = { "break-after": "avoid-column !important" };
  rules[".math-block"] = { "overflow": "visible !important" };
  rules[".reader-expandable"] = { "cursor": "zoom-in !important" };
  rules["figcaption"] = { ...rules.figcaption, "font-size": ".88em !important", "line-height": "1.5 !important" };
  rules[".reader-algorithm .algorithm-case-group"] = { "break-inside": "avoid-column !important", "page-break-inside": "avoid !important" };
  rules[".reader-algorithm td.algorithm-case"] = { "width": "3.5em !important", "min-width": "3.5em !important", "vertical-align": "middle !important", "padding": ".1em .3em !important", "border-inline-start": `.08em solid ${palette.accent} !important` };
  rules[".reader-algorithm .algorithm-case math *"] = { "color": `${palette.accent} !important` };
  rules[".reader-algorithm td.algorithm-case-empty"] = { "width": "3.5em !important" };
  rules["figure.reader-flow-figure, figure.reader-flow-figure figcaption, figure.reader-flow-figure figcaption p"] = { "break-inside": "auto !important", "page-break-inside": "auto !important" };
  rules["figure.reader-flow-figure img"] = { "break-after": "avoid-column !important", "object-fit": "contain !important", "max-width": "var(--comfortable-figure-width, 100%) !important", "max-height": "var(--comfortable-figure-height, none) !important" };
  rules["figure.reader-flow-figure figcaption"] = { "break-before": "avoid-column !important" };
  rules[".algorithm tr:first-child, .reader-algorithm tr:first-child"] = { "break-after": "avoid-column !important" };
  rules["figure"] = { "margin": ".8em 0 !important", "break-inside": "auto !important", "page-break-inside": "auto !important" };
  rules["figure img"] = { "background": "#faf8f2", "border": `.45em solid ${theme === "night" ? "#c9c3b8" : "#f3ede3"}`, "box-sizing": "border-box", "cursor": "zoom-in", "break-after": "avoid-column !important" };
  rules["pre"] = { "break-inside": "auto !important", "page-break-inside": "auto !important", "overflow": "visible !important", "box-decoration-break": "clone", "-webkit-box-decoration-break": "clone" };
  rules[".reader-scroll-math-detail"] = { "outline": `1px solid ${palette.border} !important`, "outline-offset": "2px", "cursor": "zoom-in !important" };
  rules[".reader-scroll-math-detail::after"] = { "content": '"点按放大公式 ↗"', "display": "block", "position": "sticky", "right": "0", "width": "max-content", "margin": "0 0 .2em auto", "padding": ".1em .4em", "color": `${palette.accent} !important`, "background-color": `${palette.background} !important`, "font-size": ".78em", "line-height": "1.5" };
  return rules;
}

function applyTheme(): void {
  const theme = snapshot.session.theme;
  document.documentElement.dataset.theme = theme;
  document.body.dataset.theme = theme;
  document.documentElement.style.colorScheme = (theme === "night" || theme === "contrast") ? "dark" : "light";
  shell.dataset.theme = theme;
  window.dispatchEvent(new Event('reader-theme-change'));
  const labels: Record<ThemeName, string> = {
    paper: "纸张",
    light: "明亮",
    night: "夜间",
    contrast: "高对比",
  };
  const order: ThemeName[] = ["paper", "light", "night", "contrast"];
  const nextTheme = order[(order.indexOf(theme) + 1) % order.length];
  themeCycleButton.classList.toggle("selected", theme === "night");
  themeCycleButton.dataset.currentTheme = theme;
  themeCycleButton.title = `当前：${labels[theme]}；点击切换到${labels[nextTheme]}模式`;
  themeCycleButton.setAttribute(
    "aria-label",
    `当前为${labels[theme]}模式，切换到${labels[nextTheme]}模式`,
  );
  for (const [index,runtime] of runtimes.entries()) {
    if (!runtime.rendition) continue;
    const focus=runtime.layoutAnchorCfi??runtime.cfi;
    if(focus&&contentCfiVisible(index,focus)){runtime.layoutAnchorCfi=focus;runtime.preserveSemanticFocus=true;}
    for (const contents of visibleContents(runtime.rendition)) applyReadingTheme(contents);
  }
}

function applyFontScale(): void {
  fontScaleLabel.textContent = `${readingPreferences(snapshot.session.activePane).fontScale}%`;
  runtimes.forEach((runtime, index) => {
    if (runtime.rendition || runtime.lazyPageCount > 0) void reflowPane(index, true);
  });
}

function rawResponseToBuffer(raw: ArrayBuffer | number[]): ArrayBuffer {
  if (raw instanceof ArrayBuffer) return raw;
  if (Array.isArray(raw)) return new Uint8Array(raw).buffer;
  throw new Error("书籍数据格式无效");
}

const PAGE_CANVAS_INSET = 2;
const LIVE_RESIZE_IDLE_MS = 240;
const PAGE_INFORMATION_GAIN = 1.2;

function availableStageSizeFromDimensions(
  width: number,
  height: number,
): { width: number; height: number } {
  return {
    width: Math.max(50, width - PAGE_CANVAS_INSET * 2),
    height: Math.max(50, height - PAGE_CANVAS_INSET * 2),
  };
}

function availableStageSize(stage: HTMLElement): { width: number; height: number } {
  return availableStageSizeFromDimensions(stage.clientWidth, stage.clientHeight);
}

function lazyPageEntry(page: number): string {
  return `OEBPS/images/page-${String(page).padStart(4, "0")}.svg`;
}

async function loadLazyPage(index: number, page: number): Promise<string> {
  const runtime = runtimes[index];
  const cached = runtime.lazyPageUrls.get(page);
  if (cached) return cached;
  const pending = runtime.lazyPageLoading.get(page);
  if (pending) return pending;
  if (!runtime.bookId || runtime.lazyPageCount <= 0) return "";
  const generation = runtime.generation;
  const request = invoke<ArrayBuffer | number[]>("load_book_entry", {
    bookId: runtime.bookId,
    entry: lazyPageEntry(page),
  })
    .then((raw) => {
      if (generation !== runtime.generation || runtime.lazyPageCount <= 0) return "";
      const bytes = rawResponseToBuffer(raw);
      const url = URL.createObjectURL(new Blob([bytes], { type: "image/svg+xml" }));
      runtime.lazyPageUrls.set(page, url);
      return url;
    })
    .catch((error) => {
      showToast(`第 ${page} 页加载失败：${String(error)}`, "error");
      return "";
    });
  runtime.lazyPageLoading.set(page, request);
  try {
    return await request;
  } finally {
    if (runtime.lazyPageLoading.get(page) === request) runtime.lazyPageLoading.delete(page);
  }
}

function evictLazyPages(runtime: PaneRuntime, startPage: number, visibleCount: number): void {
  const radius = Math.max(4, visibleCount * 2);
  const first = Math.max(1, startPage - radius);
  const last = Math.min(runtime.lazyPageCount, startPage + visibleCount - 1 + radius);
  for (const [page, url] of runtime.lazyPageUrls) {
    if (page < first || page > last) {
      URL.revokeObjectURL(url);
      runtime.lazyPageUrls.delete(page);
    }
  }
}

async function renderLazyPageWindow(index: number, requestedStart: number): Promise<void> {
  const runtime = runtimes[index];
  if (runtime.lazyPageCount <= 0) return;
  const pane = paneElement(index);
  const host = pane.querySelector<HTMLElement>(".epub-host");
  const canvas = host?.querySelector<HTMLElement>(".lazy-page-canvas");
  if (!host || !canvas) return;

  const visibleCount = Math.max(1, runtime.actualPageCount);
  const startPage = Math.min(runtime.lazyPageCount, Math.max(1, Math.round(requestedStart)));
  const renderGeneration = ++runtime.lazyPageGeneration;
  runtime.currentPage = startPage;
  runtime.endPage = Math.min(runtime.lazyPageCount, startPage + visibleCount - 1);
  runtime.totalPages = runtime.lazyPageCount;
  runtime.percent = runtime.lazyPageCount > 1
    ? (startPage - 1) / (runtime.lazyPageCount - 1)
    : 0;
  updatePaneProgress(index);

  const pageWidth = runtime.layoutWidth / visibleCount;
  canvas.style.setProperty("--lazy-page-width", `${pageWidth}px`);
  canvas.replaceChildren();
  const slots: Array<{ page: number; slot: HTMLElement; image: HTMLImageElement }> = [];
  for (let offset = 0; offset < visibleCount; offset += 1) {
    const page = startPage + offset;
    const slot = document.createElement("div");
    slot.className = "lazy-page-slot";
    slot.dataset.page = String(page);
    const image = document.createElement("img");
    image.alt = page <= runtime.lazyPageCount ? `第 ${page} 页` : "空白页";
    if (page <= runtime.lazyPageCount) {
      slot.append(image);
      slots.push({ page, slot, image });
    }
    canvas.append(slot);
  }

  const prefetchRadius = Math.max(4, visibleCount * 2);
  const prefetchFirst = Math.max(1, startPage - prefetchRadius);
  const prefetchLast = Math.min(
    runtime.lazyPageCount,
    startPage + visibleCount - 1 + prefetchRadius,
  );
  const pages = Array.from({ length: prefetchLast - prefetchFirst + 1 }, (_, offset) =>
    prefetchFirst + offset,
  );
  await Promise.all(pages.map((page) => loadLazyPage(index, page)));
  if (renderGeneration !== runtime.lazyPageGeneration || runtime.generation <= 0) return;
  for (const entry of slots) {
    const url = runtime.lazyPageUrls.get(entry.page);
    if (url) entry.image.src = url;
    else entry.slot.classList.add("lazy-page-error");
  }
  evictLazyPages(runtime, startPage, visibleCount);
  pane.classList.remove("loading");
  updatePaneProgress(index);
  savePaneProgress(index, 0);
  renderOverlayAnnotations(index);
}

async function openLazyPageBook(
  index: number,
  generation: number,
  pageCount: number,
  saved: BookProgress | undefined,
  stage: HTMLElement,
  host: HTMLElement,
): Promise<void> {
  const runtime = runtimes[index];
  await waitForStableStage(stage);
  if (generation !== runtime.generation) return;
  runtime.lazyPageCount = pageCount;
  runtime.rendition = null;
  runtime.book = null;
  runtime.cfi = null;
  runtime.pageMode = pageModeForBook(runtime.bookId);
  const available = availableStageSize(stage);
  runtime.actualPageCount = runtime.pageMode > 0
    ? runtime.pageMode
    : automaticPageCountForWidth(available.width,index);
  runtime.layoutWidth = available.width;
  runtime.layoutHeight = available.height;
  host.replaceChildren();
  const canvas = document.createElement("div");
  canvas.className = "lazy-page-canvas";
  host.append(canvas);
  positionRenditionCanvas(index);
  updatePaneHeader(index);
  const savedPage = Math.max(1, Number(saved?.page) || 1);
  await renderLazyPageWindow(index, savedPage);
  if (generation !== runtime.generation) return;
  paneElement(index).classList.remove("loading");
  runtime.resizeObserver = new ResizeObserver((entries) => {
    const rect = entries[0]?.contentRect;
    scheduleLiveViewportScale(
      index,
      rect?.width ?? stage.clientWidth,
      rect?.height ?? stage.clientHeight,
    );
  });
  runtime.resizeObserver.observe(stage);
  updatePaneProgress(index);
}

async function waitForStableStage(stage: HTMLElement, quietMs = 360, maxMs = 2600): Promise<void> {
  const started = performance.now();
  let stableSince = started;
  let lastWidth = stage.clientWidth;
  let lastHeight = stage.clientHeight;
  while (performance.now() - started < maxMs) {
    await new Promise<void>((resolve) => window.setTimeout(resolve, 80));
    const width = stage.clientWidth;
    const height = stage.clientHeight;
    if (Math.abs(width - lastWidth) > 1 || Math.abs(height - lastHeight) > 1) {
      lastWidth = width;
      lastHeight = height;
      stableSince = performance.now();
      continue;
    }
    if (width >= 50 && height >= 50 && performance.now() - stableSince >= quietMs) return;
  }
}

function automaticPageCountForWidth(width: number,index=snapshot.session.activePane): number {
  const idealPageWidth = 420 * Math.pow(readingPreferences(index).fontScale / 100, 0.35);
  return Math.min(10, Math.max(1, Math.floor((width + 18) / (idealPageWidth + 18))));
}

function automaticPageCount(host: HTMLElement): number {
  return automaticPageCountForWidth(host.clientWidth,Number(host.closest<HTMLElement>('.reader-pane')?.dataset.paneIndex??snapshot.session.activePane));
}

function densityPreservingFontScale(runtime: PaneRuntime, host: HTMLElement): number {
  const fontScale=readingPreferences(runtimes.indexOf(runtime)).fontScale;
  if(runtime.readingMode==='scroll'||runtime.pageMode===0)return fontScale;
  const referencePageCount = automaticPageCount(host);
  const selectedPageCount = Math.max(1, runtime.actualPageCount);
  // Page capacity is approximately area / fontSize². Scaling by the square
  // root of the page-area ratio keeps capacity stable across 1–10 pages. The
  // additional 1/sqrt(1.2) factor adds about 20% more information to every
  // logical page without making 10-page mode disproportionately sparse.
  const densityFactor = Math.sqrt(
    referencePageCount / (selectedPageCount * PAGE_INFORMATION_GAIN),
  );
  return Math.min(220, Math.max(16, fontScale * densityFactor));
}

function applyAdaptiveFontScale(index: number, host: HTMLElement): void {
  const runtime = runtimes[index];
  if (!runtime.rendition) return;
  runtime.effectiveFontScale = densityPreservingFontScale(runtime, host);
  // Absolute CSS pixels avoid Chromium clamping small percentage fonts to its
  // minimum logical font size; the density contract uses a 16px reference.
  runtime.rendition.themes.fontSize(`${18 * runtime.effectiveFontScale / 100}px`);
}

function positionRenditionCanvas(
  index: number,
  measuredStage?: { width: number; height: number },
): void {
  const runtime = runtimes[index];
  const pane = paneElement(index);
  const stage = pane.querySelector<HTMLElement>(".pane-stage");
  const host = pane.querySelector<HTMLElement>(".epub-host");
  const annotationLayer = pane.querySelector<HTMLElement>(".annotation-layer");
  if (!stage || !host || !annotationLayer || runtime.layoutWidth < 50 || runtime.layoutHeight < 50) return;

  const stageWidth = measuredStage?.width ?? stage.clientWidth;
  const stageHeight = measuredStage?.height ?? stage.clientHeight;
  const available = availableStageSizeFromDimensions(stageWidth, stageHeight);
  // Native window dragging never changes EPUB geometry. Every mode temporarily
  // keeps its logical canvas and uses a compositor-only uniform transform, so
  // the leftmost page anchor cannot move and no edge page can be cut in half.
  const scale = Math.min(
    available.width / runtime.layoutWidth,
    available.height / runtime.layoutHeight,
  );
  const safeScale = Math.max(0.05, scale);
  // Pin the first page to a constant top-left safe inset. Re-centering on every
  // resize frame makes text move vertically and horizontally while it scales,
  // which feels like trembling when the left window border is dragged.
  const offsetX = PAGE_CANVAS_INSET;
  const offsetY = PAGE_CANVAS_INSET;
  const logicalWidth = `${runtime.layoutWidth}px`;
  const logicalHeight = `${runtime.layoutHeight}px`;
  const transform = `translate3d(${offsetX}px, ${offsetY}px, 0) scale(${safeScale})`;

  // Avoid redundant layout writes in the ResizeObserver hot path. Width and
  // height change only on an intentional repagination; live dragging updates
  // one composited transform property per animation frame.
  for (const canvas of [host, annotationLayer]) {
    if (canvas.style.width !== logicalWidth) canvas.style.width = logicalWidth;
    if (canvas.style.height !== logicalHeight) canvas.style.height = logicalHeight;
    if (canvas.style.left !== "0px") canvas.style.left = "0px";
    if (canvas.style.top !== "0px") canvas.style.top = "0px";
    if (canvas.style.right !== "auto") canvas.style.right = "auto";
    if (canvas.style.bottom !== "auto") canvas.style.bottom = "auto";
    if (canvas.style.transformOrigin !== "top left") canvas.style.transformOrigin = "top left";
    if (canvas.style.transform !== transform) canvas.style.transform = transform;
  }

  runtime.viewportWidth = available.width;
  runtime.viewportHeight = available.height;
  runtime.viewportScale = safeScale;
}

function scheduleLiveViewportScale(
  index: number,
  width: number,
  height: number,
): void {
  const runtime = runtimes[index];
  if (!runtime.rendition && runtime.lazyPageCount <= 0) return;
  const pane = paneElement(index);
  if (!runtime.resizeActive) {
    runtime.resizeActive = true;
    const focus=runtime.layoutAnchorCfi;
    runtime.resizeAnchorCfi = runtime.readingMode==='scroll'
      ? focus&&runtime.preserveSemanticFocus&&contentCfiVisible(index,focus) ? focus : firstVisibleContentCfi(index)??runtime.cfi
      : runtime.pageMode===0 ? focus??runtime.cfi : runtime.cfi;
    pane.classList.add("live-resizing");
  }
  runtime.pendingStageWidth = width;
  runtime.pendingStageHeight = height;

  if (runtime.resizeFrame === null) {
    runtime.resizeFrame = window.requestAnimationFrame(() => {
      runtime.resizeFrame = null;
      positionRenditionCanvas(index, {
        width: runtime.pendingStageWidth,
        height: runtime.pendingStageHeight,
      });
    });
  }

  if (runtime.resizeTimer !== null) window.clearTimeout(runtime.resizeTimer);
  runtime.resizeTimer = window.setTimeout(() => {
    runtime.resizeTimer = null;
    if (runtime.resizeFrame !== null) {
      window.cancelAnimationFrame(runtime.resizeFrame);
      runtime.resizeFrame = null;
      positionRenditionCanvas(index, {
        width: runtime.pendingStageWidth,
        height: runtime.pendingStageHeight,
      });
    }
    const preservedAnchor = runtime.resizeAnchorCfi;
    runtime.resizeActive = false;
    if (preservedAnchor) runtime.cfi = preservedAnchor;
    runtime.resizeAnchorCfi = null;
    pane.classList.remove("live-resizing");
    updatePaneHeader(index);
    // Live dragging scales the existing canvas. Automatic mode adapts only
    // once the resize settles, restoring the text anchor through the reflow.
    if ((runtime.readingMode==='scroll'||runtime.pageMode === 0) && (Math.abs(runtime.layoutWidth - runtime.viewportWidth) > 2 || Math.abs(runtime.layoutHeight - runtime.viewportHeight) > 2)) {
      if(runtime.readingMode==='scroll')runtime.layoutAnchorCfi=preservedAnchor;
      void reflowPane(index, true);
    }
  }, LIVE_RESIZE_IDLE_MS);
}

const ATOMIC_BLOCK_SELECTOR = [
  "figure",
  "table",
  "pre",
  ".math-block",
  ".math-display",
  ".MathJax_Display",
  ".katex-display",
  "math[display='block']",
  "img",
  "svg",
  "video",
].join(", ");

function visibleContents(rendition: Rendition): EpubContents[] {
  const contents = rendition.getContents() as unknown as EpubContents | EpubContents[];
  return Array.isArray(contents) ? contents : contents ? [contents] : [];
}

function ensureBookProgress(bookId: string, paneIndex: number): BookProgress {
  const runtime = runtimes[paneIndex];
  const existing = snapshot.progress[bookId];
  if (existing) {
    if (!Array.isArray(existing.annotations)) existing.annotations = [];
    return existing;
  }
  const created: BookProgress = {
    cfi: runtime.cfi,
    page: runtime.currentPage,
    totalPages: runtime.totalPages,
    percent: runtime.percent,
    updatedAt: Math.floor(Date.now() / 1000),
    pageMode: runtime.pageMode,
    annotations: [],
  };
  snapshot.progress[bookId] = created;
  return created;
}

function annotationPage(runtime: PaneRuntime, progression: number): number {
  const total = Math.max(1, runtime.totalPages);
  return total > 1 ? 1 + Math.round(clampNumber(progression, 0, 1, 0) * (total - 1)) : 1;
}

function progressionForSlot(runtime: PaneRuntime, slot: number): number {
  const total = Math.max(1, runtime.totalPages);
  const page = Math.min(total, Math.max(1, (runtime.currentPage || 1) + slot));
  return total > 1 ? (page - 1) / (total - 1) : runtime.percent;
}

function hexToRgba(hex: string, alpha: number): string {
  const safe = safeAnnotationColor(hex).slice(1);
  const red = Number.parseInt(safe.slice(0, 2), 16);
  const green = Number.parseInt(safe.slice(2, 4), 16);
  const blue = Number.parseInt(safe.slice(4, 6), 16);
  return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
}

function setTextAnnotationStyle(span: HTMLSpanElement, annotation: TextAnnotation): void {
  span.style.fontFamily = annotation.fontFamily;
  span.style.fontSize = `${annotation.fontSize / 100}em`;
  if (annotation.highlight) span.style.backgroundColor = hexToRgba(annotation.color, 0.3);
  if (annotation.bold) span.style.fontWeight = "750";
  if (annotation.wave || annotation.underline) {
    span.style.textDecorationLine = "underline";
    span.style.textDecorationStyle = annotation.wave ? "wavy" : "solid";
    span.style.textDecorationColor = annotation.color;
    span.style.textDecorationThickness = annotation.wave ? "1.5px" : "1.2px";
    span.style.textUnderlineOffset = "0.16em";
  }
  if (annotation.box) {
    span.style.outline = `1.5px solid ${annotation.color}`;
    span.style.outlineOffset = "1px";
    span.style.borderRadius = "2px";
  }
  span.title = annotation.comment || annotation.selectedText;
}

function wrapTextRange(range: Range, annotation: TextAnnotation): void {
  const document = range.startContainer.ownerDocument;
  if (!document || range.collapsed) return;
  const common = range.commonAncestorContainer;
  const root = common.nodeType === Node.TEXT_NODE ? common.parentElement : common as Element;
  if (!root) return;
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  const nodes: Text[] = [];
  let current = walker.nextNode();
  while (current) {
    const textNode = current as Text;
    try {
      if (textNode.data.length > 0 && range.intersectsNode(textNode)) nodes.push(textNode);
    } catch {
      // Detached nodes can disappear while epub.js rotates continuous views.
    }
    current = walker.nextNode();
  }

  for (const node of nodes) {
    if (node.parentElement?.closest(`[data-reader-annotation="${CSS.escape(annotation.id)}"]`)) continue;
    let start = node === range.startContainer ? range.startOffset : 0;
    let end = node === range.endContainer ? range.endOffset : node.data.length;
    start = Math.max(0, Math.min(start, node.data.length));
    end = Math.max(start, Math.min(end, node.data.length));
    if (end <= start) continue;
    if (end < node.data.length) node.splitText(end);
    const selectedNode = start > 0 ? node.splitText(start) : node;
    const parent = selectedNode.parentNode;
    if (!parent) continue;
    const span = document.createElement("span");
    span.className = "reader-annotation";
    span.dataset.readerAnnotation = annotation.id;
    setTextAnnotationStyle(span, annotation);
    parent.insertBefore(span, selectedNode);
    span.appendChild(selectedNode);
  }
}

function applyTextAnnotations(index: number): void {
  const runtime = runtimes[index];
  const rendition = runtime.rendition;
  if (!rendition || !runtime.bookId||editionHeld(runtime.bookId)) return;
  const marks = annotationsForBook(runtime.bookId).filter(
    (annotation): annotation is TextAnnotation => annotation.kind === "text-mark",
  );
  for (const annotation of marks) {
    const alreadyApplied = visibleContents(rendition).some((contents) =>
      Array.from(contents.document.querySelectorAll<HTMLElement>("[data-reader-annotation]"))
        .some((element) => element.dataset.readerAnnotation === annotation.id),
    );
    if (alreadyApplied) continue;
    try {
      const range = rendition.getRange(annotation.cfiRange, "reader-annotation");
      if (range) { wrapTextRange(range, annotation); if(range.startContainer.ownerDocument)singleSectionPageCounts.delete(range.startContainer.ownerDocument); }
    } catch {
      // A CFI outside the currently mounted section is applied when that view renders.
    }
  }
}

function removeVisibleTextAnnotation(index: number, annotationId: string): void {
  const rendition = runtimes[index].rendition;
  if (!rendition) return;
  for (const contents of visibleContents(rendition)) {
    const spans = Array.from(contents.document.querySelectorAll<HTMLSpanElement>("[data-reader-annotation]"))
      .filter((span) => span.dataset.readerAnnotation === annotationId);
    for (const span of spans) {
      const parent = span.parentNode;
      if (!parent) continue;
      while (span.firstChild) parent.insertBefore(span.firstChild, span);
      span.remove();
      parent.normalize();
    }
  }
}


function pageAnnotationAnchor(index: number, slot: number): { anchorCfi: string | null; progression: number; anchorVersion?: number } {
  const runtime = runtimes[index];
  const anchorCfi = firstVisibleContentCfi(index, slot);
  let progression = progressionForSlot(runtime, slot);
  if (anchorCfi && runtime.book && runtime.book.locations.length() > 0) {
    const located = runtime.book.locations.percentageFromCfi(anchorCfi);
    if (typeof located === "number" && Number.isFinite(located)) progression = located;
  }
  return { anchorCfi: anchorCfi ?? runtime.cfi, progression, anchorVersion: anchorCfi ? 2 : undefined };
}

// New notes use a CFI captured from their own page, even while the global page
// index is still loading. Older records retain their original progression rule.
function annotationAnchorPoint(index:number,annotation:FreeTextAnnotation|DrawingAnnotation):{x:number;y:number}|null{
  const runtime=runtimes[index];if(!annotation.anchorCfi||!runtime.rendition)return null;
  const host=paneElement(index).querySelector<HTMLElement>('.epub-host')!;const hostRect=host.getBoundingClientRect();
  for(const contents of visibleContents(runtime.rendition)){
    try{
      if(runtime.book?.spine.get(annotation.anchorCfi)?.index!==contents.sectionIndex)continue;
      const range=contents.range(annotation.anchorCfi,'reader-annotation');const node=range.startContainer;
      if(range.collapsed&&node.nodeType===Node.TEXT_NODE&&range.startOffset<(node.textContent?.length??0))range.setEnd(node,range.startOffset+1);
      else if(range.collapsed&&node.childNodes[range.startOffset])range.selectNode(node.childNodes[range.startOffset]);
      const rect=range.getBoundingClientRect();const frame=contents.window.frameElement as HTMLIFrameElement;
      if(!frame?.clientWidth||(!rect.width&&!rect.height))continue;
      const fr=frame.getBoundingClientRect(),scale=fr.width/frame.clientWidth,logical=Math.max(.001,runtime.viewportScale);
      return{x:(fr.left+rect.left*scale-hostRect.left)/logical,y:(fr.top+rect.top*scale-hostRect.top)/logical};
    }catch{continue;}
  }return null;
}
function captureAnnotationPlacement(index:number,annotation:FreeTextAnnotation|DrawingAnnotation):void{
  const runtime=runtimes[index],point=annotationAnchorPoint(index,annotation);if(!point)return;
  const width=runtime.layoutWidth/Math.max(1,runtime.actualPageCount),slot=Math.floor((point.x+.5)/width);
  annotation.contentPlacement={mode:runtime.readingMode,anchorX:point.x-slot*width,anchorY:point.y,width,height:runtime.layoutHeight};
}
function visibleAnnotationSlot(index: number, annotation: FreeTextAnnotation | DrawingAnnotation): number | null {
  const runtime = runtimes[index];
  if(runtime.readingMode==='scroll')return annotationAnchorPoint(index,annotation)?0:null;
  if (annotation.anchorVersion !== 2 || !annotation.anchorCfi || !runtime.book || !runtime.rendition) {
    return annotationPage(runtime, annotation.progression) - Math.max(1, runtime.currentPage || 1);
  }
  const section = runtime.book.spine.get(annotation.anchorCfi);
  const host = paneElement(index).querySelector<HTMLElement>(".epub-host");
  if (!section || !host) return null;
  const hostRect = host.getBoundingClientRect();
  const pageWidth = hostRect.width / Math.max(1, runtime.actualPageCount);
  for (const contents of visibleContents(runtime.rendition)) {
    if (contents.sectionIndex !== section.index) continue;
    try {
      const frame = contents.window.frameElement as HTMLIFrameElement | null;
      if (!frame || !frame.clientWidth) continue;
      const range = contents.range(annotation.anchorCfi, "reader-annotation");
      const node = range.startContainer;
      if (range.collapsed && node.nodeType === Node.TEXT_NODE && range.startOffset < (node.textContent?.length ?? 0)) range.setEnd(node, range.startOffset + 1);
      else if (range.collapsed && node.nodeType === Node.ELEMENT_NODE && node.childNodes[range.startOffset]) range.selectNode(node.childNodes[range.startOffset]);
      let rect = range.getBoundingClientRect();
      if (rect.width < 0.1 && rect.height < 0.1) {
        const parent = node.nodeType === Node.ELEMENT_NODE ? node as Element : node.parentElement;
        const image = parent?.matches("img,svg,video") ? parent : parent?.querySelector("img,svg,video");
        if (image) { range.selectNode(image); rect = range.getBoundingClientRect(); }
      }
      const frameRect = frame.getBoundingClientRect();
      const screenX = frameRect.left + rect.left * frameRect.width / frame.clientWidth;
      return Math.floor((screenX - hostRect.left + 0.5) / pageWidth);
    } catch { continue; }
  }
  return null;
}

function annotationOverlay(index: number): HTMLElement | null {
  return paneElement(index).querySelector<HTMLElement>(".annotation-layer");
}

function renderOverlayAnnotations(index: number): void {
  const runtime = runtimes[index];
  const overlay = annotationOverlay(index);
  if (!overlay) return;
  const drawingLayer = overlay.querySelector<SVGSVGElement>(".drawing-layer");
  const noteLayer = overlay.querySelector<HTMLElement>(".free-note-layer");
  if (!drawingLayer || !noteLayer || !runtime.bookId || runtime.layoutWidth < 50||editionHeld(runtime.bookId)) {
    overlay.classList.remove("interactive");overlay.dataset.tool="read";
    drawingLayer?.replaceChildren();
    noteLayer?.replaceChildren();
    return;
  }

  const pageCount = Math.max(1, runtime.actualPageCount);
  const pageWidth = runtime.layoutWidth / pageCount;
  drawingLayer.setAttribute("viewBox", `0 0 ${runtime.layoutWidth} ${runtime.layoutHeight}`);
  drawingLayer.setAttribute("preserveAspectRatio", "none");

  const drawingMarkup: string[] = [];
  const noteMarkup: string[] = [];
  const annotations=annotationsForBook(runtime.bookId);
  let migratedGeometry=false;
  const bounds=new Map<string,{minX:number;maxX:number;minY:number;maxY:number}>();
  for(const annotation of annotations){
    if(annotation.kind!=="drawing")continue;
    const geometry=annotation.inkGeometry;
    if(!geometry||!Number.isFinite(geometry.columnWidth)||geometry.columnWidth<=0||!Number.isFinite(geometry.pageHeight)||geometry.pageHeight<=0){
      annotation.inkGeometry={columnWidth:pageWidth,pageHeight:runtime.layoutHeight,group:`legacy-${annotation.anchorCfi??annotation.progression}`,basis:"legacy-current-layout"};
      migratedGeometry=true;
    }
    const key=annotation.inkGeometry!.group;const box=bounds.get(key)??{minX:Infinity,maxX:-Infinity,minY:Infinity,maxY:-Infinity};
    for(const point of annotation.points){box.minX=Math.min(box.minX,point.x);box.maxX=Math.max(box.maxX,point.x);box.minY=Math.min(box.minY,point.y);box.maxY=Math.max(box.maxY,point.y);}bounds.set(key,box);
  }
  for (const annotation of annotations) {
    if (annotation.kind === "text-mark") continue;
    const slot = visibleAnnotationSlot(index, annotation);
    if (slot === null || slot < 0 || slot >= pageCount) continue;
    const placement=annotation.contentPlacement;
    const followsContent=runtime.readingMode==='scroll'||placement?.mode==='scroll';
    const anchor=followsContent?annotationAnchorPoint(index,annotation):null;
    const positionScale=placement?Math.min(pageWidth/placement.width,runtime.layoutHeight/placement.height):1;
    const contentX=anchor&&placement?anchor.x-placement.anchorX*positionScale:slot*pageWidth;
    const contentY=anchor?anchor.y-(placement?.anchorY??0)*positionScale:0;
    if (annotation.kind === "drawing") {
      const geometry=annotation.inkGeometry!;const box=bounds.get(geometry.group)!;
      const centerX=(box.minX+box.maxX)/2,centerY=(box.minY+box.maxY)/2;
      const inkScale=Math.min(pageWidth/geometry.columnWidth,runtime.layoutHeight/geometry.pageHeight);
      const path = annotation.points
        .map((point, pointIndex) => {
          const x = followsContent?contentX+point.x*geometry.columnWidth*inkScale:slot * pageWidth + centerX*pageWidth+(point.x-centerX)*geometry.columnWidth*inkScale;
          const y = followsContent?contentY+point.y*geometry.pageHeight*inkScale:centerY*runtime.layoutHeight+(point.y-centerY)*geometry.pageHeight*inkScale;
          return `${pointIndex === 0 ? "M" : "L"}${x.toFixed(2)} ${y.toFixed(2)}`;
        })
        .join(" ");
      drawingMarkup.push(
        `<path class="saved-drawing" data-annotation-id="${escapeHtml(annotation.id)}" d="${path}" stroke="${annotation.color}" stroke-width="${Math.max(.65,annotation.strokeWidth*inkScale)}" vector-effect="non-scaling-stroke" fill="none" stroke-linecap="round" stroke-linejoin="round" />`,
      );
      continue;
    }
    const left = followsContent?contentX+annotation.x*(placement?.width??pageWidth)*positionScale:slot * pageWidth + annotation.x * pageWidth;
    const top = followsContent?contentY+annotation.y*(placement?.height??runtime.layoutHeight)*positionScale:annotation.y * runtime.layoutHeight;
    if(runtime.readingMode==='scroll'&&(top>=runtime.layoutHeight||top < -runtime.layoutHeight))continue;
    const classes = [
      "free-note",
      annotation.bold ? "note-bold" : "",
      annotation.underline ? "note-underline" : "",
      annotation.wave ? "note-wave" : "",
      annotation.box ? "note-box" : "",
    ].filter(Boolean).join(" ");
    noteMarkup.push(`
      <article class="${classes}" data-annotation-id="${escapeHtml(annotation.id)}"
        style="left:${left}px;top:${top}px;--note-color:${annotation.color};--note-background:${hexToRgba(annotation.color, annotation.highlight ? 0.22 : 0.09)};font-family:${annotation.fontFamily};font-size:${(14 * annotation.fontSize / 100).toFixed(2)}px;max-width:${Math.min(runtime.layoutWidth-16,Math.max(180, pageWidth * 0.76)).toFixed(1)}px;max-height:${Math.max(80,runtime.layoutHeight-16)}px;overflow:auto">
        <button class="free-note-drag" type="button" title="拖动笔记" aria-label="拖动笔记">⋮⋮</button>
        <div class="free-note-content" contenteditable="true" spellcheck="false" data-placeholder="在这里写笔记">${escapeHtml(annotation.text)}</div>
        <button class="free-note-delete" type="button" title="删除笔记" aria-label="删除笔记">×</button>
      </article>`);
  }
  drawingLayer.innerHTML = drawingMarkup.join("");
  noteLayer.innerHTML = noteMarkup.join("");
  for(const note of Array.from(noteLayer.querySelectorAll<HTMLElement>(".free-note"))){
    note.style.left=`${Math.max(4,Math.min(parseFloat(note.style.left),runtime.layoutWidth-note.offsetWidth-8))}px`;
    if(runtime.readingMode!=='scroll')note.style.top=`${Math.max(4,Math.min(parseFloat(note.style.top),runtime.layoutHeight-note.offsetHeight-8))}px`;
  }
  if(migratedGeometry)savePaneProgress(index);
  overlay.dataset.tool = annotationTool;
  overlay.classList.toggle("interactive", annotationTool !== "read");
}

function renderAnnotationPanel(): void {
  const index = snapshot.session.activePane;
  const runtime = runtimes[index];
  const book = getBook(runtime.bookId);
  annotationBookTitle.textContent = (book?.title ?? "请先打开一本书")+(editionHeld(runtime.bookId)?" · 旧版批注待核对":"");
  const pending = pendingSelections[index];
  const applyButton=app.querySelector<HTMLButtonElement>('.apply-text-mark');if(applyButton)applyButton.disabled=!pending?.text||editionHeld(runtime.bookId);
  selectionStatus.textContent = editionHeld(runtime.bookId)?"原位置、文字与笔迹完整保留；恢复原 EPUB 后重开，或核对新版的段落对应关系。":pending
    ? `已选 ${pending.text.length} 字：${pending.text.slice(0, 46)}${pending.text.length > 46 ? "…" : ""}`
    : "在原文中拖选文字，再设置格式";
  const annotations = annotationsForBook(runtime.bookId);
  if (!book) {
    annotationList.innerHTML = '<div class="annotation-list-empty">当前窗格还没有打开书</div>';
    return;
  }
  if (annotations.length === 0) {
    annotationList.innerHTML = '<div class="annotation-list-empty">本书还没有笔记或标记</div>';
    return;
  }
  annotationList.innerHTML = annotations
    .slice()
    .sort((left, right) => right.createdAt - left.createdAt)
    .map((annotation) => {
      const label = annotation.kind === "text-mark" ? "原文" : annotation.kind === "free-text" ? "文字" : "涂鸦";
      const preview = annotation.kind === "text-mark"
        ? annotation.comment || annotation.selectedText
        : annotation.kind === "free-text"
          ? annotation.text || "空白文字笔记"
          : `自由涂鸦 · ${annotation.points.length} 个笔迹点`;
      return `<article class="annotation-list-item" data-annotation-id="${escapeHtml(annotation.id)}">
        <button class="annotation-jump" type="button" title="跳到这条笔记"><span style="--item-color:${annotation.color}">${label}</span><strong>${escapeHtml(preview.slice(0, 80))}</strong></button>
        <button class="annotation-delete" type="button" title="删除这条笔记" aria-label="删除这条笔记">${icons.trash}</button>
      </article>`;
    })
    .join("");
}

function setAnnotationPanelVisible(visible: boolean): void {
  if(visible){closeDrawer();closeBookNavigation();setReadingSettings(false);workspace?.showPanel(false,false);}
  const wasVisible = annotationPanel.classList.contains("visible");
  const hadFocus = annotationPanel.contains(document.activeElement);
  annotationPanel.inert = !visible;
  annotationPanel.classList.toggle("visible", visible);
  annotationPanel.setAttribute("aria-hidden", String(!visible));
  annotationToggle.classList.toggle("selected", visible);
  annotationToggle.setAttribute("aria-expanded", String(visible));
  if (visible) {
    renderAnnotationPanel();
    if (!wasVisible) requireElement<HTMLButtonElement>(annotationPanel, ".annotation-close").focus();
  } else {
    if(annotationTool!=='read')setAnnotationTool('read');
    if(wasVisible&&hadFocus)(toolsVisible?annotationToggle:app.querySelector<HTMLButtonElement>('.tools-toggle'))?.focus({preventScroll:true});
  }
}

function setAnnotationTool(tool: AnnotationTool): void {
  if(tool==="pen"&&annotationTool!=="pen")inkSession=crypto.randomUUID();
  annotationTool = tool;
  app.querySelectorAll<HTMLButtonElement>(".annotation-tool").forEach((button) => {
    const selected = button.dataset.tool === tool;
    button.classList.toggle("selected", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
  runtimes.forEach((runtime, index) => {
    if (runtime.bookId) renderOverlayAnnotations(index);
  });
}

function persistBookAnnotations(bookId: string, delay = 0): void {
  const paneIndex = runtimes.findIndex((runtime) => runtime.bookId === bookId);
  if (paneIndex >= 0) savePaneProgress(paneIndex, delay);
  if (delay === 0) renderLibrary(searchInput.value);
}

function syncBookAnnotations(bookId: string): void {
  runtimes.forEach((runtime, index) => {
    if (runtime.bookId !== bookId) return;
    applyTextAnnotations(index);
    renderOverlayAnnotations(index);
  });
  renderAnnotationPanel();
  persistBookAnnotations(bookId);
}

function addBookAnnotation(index: number, annotation: ReaderAnnotation): void {
  const runtime = runtimes[index];
  if(!annotationEditionAvailable(runtime.bookId))return;
  if (!runtime.bookId) return;
  ensureBookProgress(runtime.bookId, index).annotations.push(annotation);
  syncBookAnnotations(runtime.bookId);
}

function deleteBookAnnotation(bookId: string, annotationId: string): void {
  if(!annotationEditionAvailable(bookId))return;
  const annotations = annotationsForBook(bookId);
  const position = annotations.findIndex((annotation) => annotation.id === annotationId);
  if (position < 0) return;
  runtimes.forEach((runtime, index) => {
    if (runtime.bookId === bookId) removeVisibleTextAnnotation(index, annotationId);
  });
  annotations.splice(position, 1);
  syncBookAnnotations(bookId);
}

function undoLastAnnotation(): void {
  const runtime = runtimes[snapshot.session.activePane];
  if(!annotationEditionAvailable(runtime.bookId))return;
  if (!runtime.bookId) return;
  const latest = annotationsForBook(runtime.bookId)
    .slice()
    .sort((left, right) => right.createdAt - left.createdAt)[0];
  if (!latest) {
    showToast("这本书还没有可撤销的笔记");
    return;
  }
  deleteBookAnnotation(runtime.bookId, latest.id);
  showToast("已撤销最后一条笔记");
}

function handleTextSelection(index: number, cfiRange: string, contents: EpubContents): void {
  const selection = contents.window.getSelection();
  const text = selection?.toString().trim() ?? "";
  if (!text) return;
  let stableCfi = cfiRange;
  try {
    if (selection && selection.rangeCount > 0) {
      stableCfi = contents.cfiFromRange(selection.getRangeAt(0), "reader-annotation");
    }
  } catch {
    // The cfiRange emitted by epub.js is still a valid fallback.
  }
  pendingSelections[index] = { cfiRange: stableCfi, text, contents };
  setActivePane(index);
  // Selecting text must remain a quiet reading action. Keep the selection so
  // the user can explicitly open Notes later, but never reveal either overlay
  // just because epub.js emitted a selection event.
  if (annotationPanel.classList.contains("visible")) renderAnnotationPanel();
}

function applyPendingTextMark(): void {
  const index = snapshot.session.activePane;
  const runtime = runtimes[index];
  if(!annotationEditionAvailable(runtime.bookId))return;
  const pending = pendingSelections[index];
  if (!runtime.bookId || !pending) {
    showToast("请先在原文中拖选一段文字", "error");
    return;
  }
  const style = currentAnnotationStyle();
  if (!style.highlight && !style.bold && !style.underline && !style.wave && !style.box) {
    showToast("请至少选择一种原文格式", "error");
    return;
  }
  const annotation: TextAnnotation = {
    ...style,
    id: crypto.randomUUID(),
    kind: "text-mark",
    cfiRange: pending.cfiRange,
    selectedText: pending.text,
    comment: selectionComment.value.trim(),
    createdAt: Date.now(),
  };
  pending.contents.window.getSelection()?.removeAllRanges();
  pendingSelections[index] = null;
  selectionComment.value = "";
  addBookAnnotation(index, annotation);
  showToast("原文标记已保存到这本书");
}

function jumpToAnnotation(annotationId: string): void {
  const runtime = runtimes[snapshot.session.activePane];
  if(!annotationEditionAvailable(runtime.bookId))return;
  if (!runtime.bookId || !runtime.rendition) return;
  const annotation = annotationsForBook(runtime.bookId).find((item) => item.id === annotationId);
  if (!annotation) return;
  const target = annotation.kind === "text-mark" ? annotation.cfiRange : annotation.anchorCfi;
  if (!target) return;
  void jumpToBookTarget(target);
}

function pointOnAnnotationLayer(event: PointerEvent, index: number): {
  slot: number;
  x: number;
  y: number;
  logicalX: number;
  logicalY: number;
} | null {
  const runtime = runtimes[index];
  const overlay = annotationOverlay(index);
  if (!overlay || runtime.layoutWidth < 50 || runtime.layoutHeight < 50) return null;
  const rect = overlay.getBoundingClientRect();
  if (rect.width < 1 || rect.height < 1) return null;
  const logicalX = (event.clientX - rect.left) / rect.width * runtime.layoutWidth;
  const logicalY = (event.clientY - rect.top) / rect.height * runtime.layoutHeight;
  const pageWidth = runtime.layoutWidth / Math.max(1, runtime.actualPageCount);
  const slot = Math.min(
    Math.max(1, runtime.actualPageCount) - 1,
    Math.max(0, Math.floor(logicalX / pageWidth)),
  );
  return {
    slot,
    x: (logicalX - slot * pageWidth) / pageWidth,
    y: logicalY / runtime.layoutHeight,
    logicalX,
    logicalY,
  };
}

function addFreeTextAt(index: number, point: { slot: number; x: number; y: number }): void {
  const runtime = runtimes[index];
  if (!runtime.bookId) return;
  const style = currentAnnotationStyle();
  const annotation: FreeTextAnnotation = {
    ...style,
    id: crypto.randomUUID(),
    kind: "free-text",
    ...pageAnnotationAnchor(index, point.slot),
    x: Math.min(0.92, Math.max(0, point.x)),
    y: Math.min(0.94, Math.max(0, point.y)),
    text: "新笔记",
    createdAt: Date.now(),
  };
  captureAnnotationPlacement(index,annotation);
  addBookAnnotation(index, annotation);
  window.requestAnimationFrame(() => {
    const content = paneElement(index).querySelector<HTMLElement>(
      `.free-note[data-annotation-id="${CSS.escape(annotation.id)}"] .free-note-content`,
    );
    if (!content) return;
    content.focus();
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(content);
    selection?.removeAllRanges();
    selection?.addRange(range);
  });
}

let inkSession=crypto.randomUUID();
function inkGroupFor(index:number,slot:number):string{
  const runtime=runtimes[index];const anchor=pageAnnotationAnchor(index,slot).anchorCfi??String(slot);
  return `${inkSession}|${runtime.bookId}|${anchor}|${runtime.layoutWidth/runtime.actualPageCount}|${runtime.layoutHeight}`;
}

function beginDrawing(index: number, event: PointerEvent): void {
  const runtime = runtimes[index];
  if(!annotationEditionAvailable(runtime.bookId))return;
  const point = pointOnAnnotationLayer(event, index);
  const overlay = annotationOverlay(index);
  if (!runtime.bookId || !point || !overlay) return;
  const annotation: DrawingAnnotation = {
    id: crypto.randomUUID(),
    kind: "drawing",
    ...pageAnnotationAnchor(index, point.slot),
    color: safeAnnotationColor(annotationColorInput.value),
    strokeWidth: clampNumber(penWidthInput.value, 1, 10, 3),
    points: [{ x: point.x, y: point.y }, { x: point.x, y: point.y }],
    inkGeometry: { columnWidth:runtime.layoutWidth/Math.max(1,runtime.actualPageCount),pageHeight:runtime.layoutHeight,group:inkGroupFor(index,point.slot),basis:"creation" },
    createdAt: Date.now(),
  };
  captureAnnotationPlacement(index,annotation);
  ensureBookProgress(runtime.bookId, index).annotations.push(annotation);
  drawingDraft = { paneIndex: index, pointerId: event.pointerId, annotation };
  overlay.setPointerCapture(event.pointerId);
  renderOverlayAnnotations(index);
}

function extendDrawing(event: PointerEvent): void {
  const draft = drawingDraft;
  if (!draft || event.pointerId !== draft.pointerId) return;
  const runtime = runtimes[draft.paneIndex];
  const point = pointOnAnnotationLayer(event, draft.paneIndex);
  if (!point) return;
  const slot = visibleAnnotationSlot(draft.paneIndex, draft.annotation) ?? 0;
  const relativeX = point.x + point.slot - slot;
  const previous = draft.annotation.points[draft.annotation.points.length - 1];
  const pageWidth = runtime.layoutWidth / Math.max(1, runtime.actualPageCount);
  const distance = Math.hypot(
    (relativeX - previous.x) * pageWidth,
    (point.y - previous.y) * runtime.layoutHeight,
  );
  if (distance < 1.5) return;
  draft.annotation.points.push({ x: relativeX, y: point.y });
  if (draft.annotation.points.length > 8000) draft.annotation.points.shift();
  if (drawingFrame !== null) return;
  drawingFrame = window.requestAnimationFrame(() => {
    drawingFrame = null;
    renderOverlayAnnotations(draft.paneIndex);
  });
}

function finishDrawing(event: PointerEvent): void {
  const draft = drawingDraft;
  if (!draft || event.pointerId !== draft.pointerId) return;
  const runtime = runtimes[draft.paneIndex];
  annotationOverlay(draft.paneIndex)?.releasePointerCapture(event.pointerId);
  drawingDraft = null;
  if (drawingFrame !== null) {
    window.cancelAnimationFrame(drawingFrame);
    drawingFrame = null;
  }
  if (runtime.bookId) syncBookAnnotations(runtime.bookId);
}

function beginNoteDrag(index: number, annotationId: string, event: PointerEvent): void {
  const runtime = runtimes[index];
  const point = pointOnAnnotationLayer(event, index);
  const annotation = annotationsForBook(runtime.bookId).find(
    (item): item is FreeTextAnnotation => item.id === annotationId && item.kind === "free-text",
  );
  if (!point || !annotation) return;
  const pageWidth = runtime.layoutWidth / Math.max(1, runtime.actualPageCount);
  const slot = visibleAnnotationSlot(index, annotation) ?? 0;
  const element=paneElement(index).querySelector<HTMLElement>(`.free-note[data-annotation-id="${CSS.escape(annotationId)}"]`);
  noteDragState = {
    paneIndex: index,
    annotationId,
    pointerId: event.pointerId,
    offsetX: point.logicalX - (element?parseFloat(element.style.left):slot * pageWidth + annotation.x * pageWidth),
    offsetY: point.logicalY - (element?parseFloat(element.style.top):annotation.y * runtime.layoutHeight),
  };
}

function moveNote(event: PointerEvent): void {
  const drag = noteDragState;
  if (!drag || event.pointerId !== drag.pointerId) return;
  const runtime = runtimes[drag.paneIndex];
  const point = pointOnAnnotationLayer(event, drag.paneIndex);
  const annotation = annotationsForBook(runtime.bookId).find(
    (item): item is FreeTextAnnotation => item.id === drag.annotationId && item.kind === "free-text",
  );
  if (!point || !annotation) return;
  const pageCount = Math.max(1, runtime.actualPageCount);
  const pageWidth = runtime.layoutWidth / pageCount;
  const left = Math.min(runtime.layoutWidth - 24, Math.max(0, point.logicalX - drag.offsetX));
  const top = Math.min(runtime.layoutHeight - 24, Math.max(0, point.logicalY - drag.offsetY));
  const slot = Math.min(pageCount - 1, Math.max(0, Math.floor(left / pageWidth)));
  if (annotation.anchorVersion !== 2 || visibleAnnotationSlot(drag.paneIndex, annotation) !== slot) {
    Object.assign(annotation, pageAnnotationAnchor(drag.paneIndex, slot));
  }
  annotation.x = (left - slot * pageWidth) / pageWidth;
  annotation.y = top / runtime.layoutHeight;
  if(runtime.readingMode==='scroll'){Object.assign(annotation,pageAnnotationAnchor(drag.paneIndex,slot));captureAnnotationPlacement(drag.paneIndex,annotation);}
  if (drawingFrame !== null) return;
  drawingFrame = window.requestAnimationFrame(() => {
    drawingFrame = null;
    renderOverlayAnnotations(drag.paneIndex);
  });
}

function finishNoteDrag(event: PointerEvent): void {
  const drag = noteDragState;
  if (!drag || event.pointerId !== drag.pointerId) return;
  const runtime = runtimes[drag.paneIndex];
  noteDragState = null;
  if (drawingFrame !== null) {
    window.cancelAnimationFrame(drawingFrame);
    drawingFrame = null;
    renderOverlayAnnotations(drag.paneIndex);
  }
  if (runtime.bookId) syncBookAnnotations(runtime.bookId);
}

function handleAnnotationPointerDown(event: PointerEvent): void {
  const target = event.target as Element;
  const overlay = target.closest<HTMLElement>(".annotation-layer");
  if (!overlay) return;
  const index = Number(overlay.dataset.paneIndex);
  if (!Number.isFinite(index)) return;
  const runtime = runtimes[index];
  if (!runtime.bookId) return;
  setActivePane(index);

  const annotationElement = target.closest<HTMLElement>("[data-annotation-id]");
  const annotationId = annotationElement?.dataset.annotationId;
  if (target.closest(".free-note-delete") && annotationId) {
    event.preventDefault();
    event.stopPropagation();
    deleteBookAnnotation(runtime.bookId, annotationId);
    return;
  }
  if (target.closest(".free-note-drag") && annotationId) {
    event.preventDefault();
    event.stopPropagation();
    beginNoteDrag(index, annotationId, event);
    return;
  }
  if (annotationTool === "eraser" && annotationId) {
    event.preventDefault();
    event.stopPropagation();
    deleteBookAnnotation(runtime.bookId, annotationId);
    return;
  }
  if (target.closest(".free-note-content")) return;
  const point = pointOnAnnotationLayer(event, index);
  if (!point) return;
  if (annotationTool === "text") {
    event.preventDefault();
    addFreeTextAt(index, point);
    return;
  }
  if (annotationTool === "pen") {
    event.preventDefault();
    beginDrawing(index, event);
  }
}

function updateFreeTextFromEditor(editor: HTMLElement): void {
  const item = editor.closest<HTMLElement>(".free-note");
  const pane = editor.closest<HTMLElement>(".reader-pane");
  const index = Number(pane?.dataset.paneIndex);
  const annotationId = item?.dataset.annotationId;
  const runtime = runtimes[index];
  if (!Number.isFinite(index) || !annotationId || !runtime?.bookId) return;
  const annotation = annotationsForBook(runtime.bookId).find(
    (entry): entry is FreeTextAnnotation => entry.id === annotationId && entry.kind === "free-text",
  );
  if (!annotation) return;
  annotation.text = (editor.innerText || "").slice(0, 24000);
  persistBookAnnotations(runtime.bookId, 320);
  renderAnnotationPanel();
}

function fitAtomicBlocks(index: number): void {
  const runtime = runtimes[index];
  const rendition = runtime.rendition;
  if (!rendition) return;
  for(const contents of visibleContents(rendition)){fitInlineStops(contents.document);for(const th of contents.document.querySelectorAll<HTMLElement>('table:not(.reader-algorithm) th'))th.classList.toggle('reader-short-table-header',Array.from(th.textContent?.trim()??'').length<=10&&!th.querySelector('math'));for(const table of contents.document.querySelectorAll<HTMLTableElement>('table'))if(runtime.readingMode==='scroll'&&table.scrollWidth>table.clientWidth+2){table.tabIndex=0;table.setAttribute('aria-label','表格，可用左右方向键或横向滚动查看全部列');}}
  if(runtime.readingMode==='scroll'){
    for(const contents of visibleContents(rendition)){
      for(const element of contents.document.querySelectorAll<HTMLElement>('.math-block,.math-display,.MathJax_Display,.katex-display')){
        if(!element.clientWidth)continue;
        if(element.scrollWidth>element.clientWidth+2){
          element.dataset.readerDetail='true';element.dataset.comfortableMathDetail='true';
          element.classList.add('reader-expandable','reader-scroll-math-detail');
          element.tabIndex=0;element.title='点按放大阅读公式，也可在此横向滚动';
          element.setAttribute('aria-description','可横向滚动；按回车或点按可放大、选择并复制公式');
        }else if(element.dataset.comfortableMathDetail==='true'){
          delete element.dataset.readerDetail;delete element.dataset.comfortableMathDetail;
          element.classList.remove('reader-expandable','reader-scroll-math-detail');
          element.removeAttribute('tabindex');element.removeAttribute('aria-description');
          if(element.title==='点按放大阅读公式，也可在此横向滚动')element.removeAttribute('title');
        }
      }
    }
    return;
  }
  for(const contents of visibleContents(rendition)){
    for(const element of contents.document.querySelectorAll<HTMLElement>('[data-comfortable-math-detail]')){
      delete element.dataset.readerDetail;delete element.dataset.comfortableMathDetail;delete element.dataset.comfortableFitKey;
      element.classList.remove('reader-expandable','reader-scroll-math-detail');
      element.removeAttribute('tabindex');element.removeAttribute('aria-description');
      if(element.title==='点按放大阅读公式，也可在此横向滚动')element.removeAttribute('title');
    }
  }
  const layout = (rendition as InternalRendition)._layout;
  const usableHeight = Math.max(80, layout.height - 44);
  const usableWidth = Math.max(24, layout.columnWidth - 8);
  const fitKey = [
    Math.round(usableWidth * 10) / 10,
    Math.round(usableHeight),
    Math.round(runtime.effectiveFontScale * 10) / 10,
    runtime.actualPageCount,
    readingPreferences(index).readerFont,
  ].join(":");

  for (const contents of visibleContents(rendition)) {
    const documentElement = contents.document.documentElement as HTMLElement;
    documentElement.style.setProperty("--comfortable-page-height", `${usableHeight}px`);
    const elements = contents.document.querySelectorAll<HTMLElement>(ATOMIC_BLOCK_SELECTOR);
    for (const element of elements) {
      // Algorithms retain row numbers and indentation across columns; shrinking
      // a 60-line procedure to fit one page makes the procedure unreadable.
      if (element.matches(".algorithm, .reader-algorithm, .algorithm-cost")) { fitAlgorithmPreview(element, fitKey, usableWidth, layout.columnWidth); continue; }
      if (element.matches("pre")) { fitFlowCodePreview(element, fitKey, usableWidth); continue; }
      if (element.matches('table') && fitFlowTablePreview(element as HTMLTableElement, fitKey, usableWidth, usableHeight, layout.columnWidth)) continue;
      // A figure owns its image; fitting both would shrink the same visual twice.
      const owner = element.parentElement?.closest(ATOMIC_BLOCK_SELECTOR);
      if (owner && !(owner.matches("figure.reader-flow-figure") && !element.matches("img, svg, video"))) continue;
      if (element.dataset.comfortableFitKey === fitKey) continue;

      element.style.removeProperty("zoom");
      element.style.removeProperty("max-height");
      element.style.removeProperty("object-fit");
      element.style.removeProperty("transform-origin");
      element.style.removeProperty("margin-left");
      element.style.removeProperty("margin-right");
      element.dataset.comfortableFitKey = fitKey;

      if (element.matches("figure") && element.querySelector("figcaption")?.textContent?.trim()) {
        // Only the image is a compact column preview. The caption remains
        // ordinary text and can continue in the next column when necessary.
        element.classList.add("reader-flow-figure");
        element.style.setProperty("--comfortable-figure-width", `${usableWidth}px`);
        element.style.setProperty("--comfortable-figure-height", `${Math.max(70, usableHeight * 0.66)}px`);
        for (const picture of Array.from(element.querySelectorAll<HTMLElement>("img"))) picture.style.removeProperty("max-height");
        continue;
      }
      const rect = element.getBoundingClientRect();
      const childRects = Array.from(element.querySelectorAll("math, math mtable, math mtd, math mtext, math mi, math mn, math mo, img, svg")).map(child => child.getBoundingClientRect());
      // Centered MathML can overflow to both sides; scrollWidth counts only
      // the right side. Include the actual child bounds before fitting.
      const left = Math.min(rect.left, ...childRects.map(child => child.left));
      const right = Math.max(rect.right, ...childRects.map(child => child.right));
      const top = Math.min(rect.top, ...childRects.map(child => child.top));
      const bottom = Math.max(rect.bottom, ...childRects.map(child => child.bottom));
      const naturalHeight = Math.max(bottom - top, element.scrollHeight);
      const naturalWidth = Math.max(right - left, element.scrollWidth);
      if (naturalHeight <= usableHeight && naturalWidth <= usableWidth) continue;

      const scale = Math.min(
        1,
        usableHeight / Math.max(1, naturalHeight),
        usableWidth / Math.max(1, naturalWidth),
      );
      if (scale < 0.88 && element.matches("table, .math-block, .math-display, math[display='block']")) {
        element.dataset.readerDetail = "true";
        element.classList.add("reader-expandable");
        element.tabIndex = 0;
        element.title = element.matches("table") ? "点按放大阅读表格" : "点按放大阅读公式";
      }
      if (scale >= 0.995) continue;

      if (element.matches("img, svg, video")) {
        element.style.setProperty("max-height", `${usableHeight}px`, "important");
        element.style.setProperty("object-fit", "contain", "important");
      } else {
        // Chromium/WebView2 zoom changes both painting and layout dimensions,
        // unlike transform, so the following column starts after the scaled block.
        element.style.setProperty("zoom", String(Math.max(0.05, scale)));
        element.style.setProperty("transform-origin", "top left");
        element.style.setProperty("margin-left", "auto");
        element.style.setProperty("margin-right", "auto");
      }
    }
  }
}

function scheduleAtomicFit(index: number): void {
  const runtime = runtimes[index];
  if (runtime.atomicFitFrame !== null) window.cancelAnimationFrame(runtime.atomicFitFrame);
  runtime.atomicFitFrame = window.requestAnimationFrame(() => {
    runtime.atomicFitFrame = window.requestAnimationFrame(() => {
      runtime.atomicFitFrame = null;
      fitAtomicBlocks(index);
      void runtime.rendition?.reportLocation();
    });
  });
}

function linearSectionCount(book: EpubBook): number {
  const sections = (book.spine as InternalSpine).spineItems ?? [];
  return sections.filter((section) => section.linear !== false).length;
}

function scheduleDynamicPageMap(index: number, delay = 260, force = false): void {
  const runtime = runtimes[index];
  if (!runtime.book || !runtime.rendition) return;
  // The source build measured these CFIs against the exact streamed XHTML.
  // Generating locations here would fetch unselected chapters.
  if(runtime.bookId&&portableBooks.has(runtime.bookId)){void runtime.rendition.reportLocation();return;}
  // Single-section books have exact local page counts. Multi-section books
  // use stable text locations, never a guessed total number of screen pages.
  if (runtime.readingMode!=='scroll' && linearSectionCount(runtime.book) === 1 && !force) {
    void runtime.rendition.reportLocation();
    return;
  }
  if (indexingBooks.has(runtime.book) || (!force && runtime.book.locations.length() > 0)) return;
  runtime.paginationGeneration += 1;
  const generation = runtime.paginationGeneration;
  if (runtime.paginationTimer !== null) window.clearTimeout(runtime.paginationTimer);
  runtime.paginationTimer = window.setTimeout(() => {
    runtime.paginationTimer = null;
    const book = runtime.book;
    const rendition = runtime.rendition;
    if (!book || !rendition || generation !== runtime.paginationGeneration) return;
    const LocationsConstructor = book.locations.constructor as unknown as new (
      spine: EpubBook["spine"],
      request: EpubBook["load"],
      pause?: number,
    ) => EpubBook["locations"];
    const locations = new LocationsConstructor(book.spine, book.load.bind(book), 1);
    indexingBooks.add(book);
    void locations
      .generate(1000)
      .then(() => {
        if (
          generation !== runtime.paginationGeneration ||
          runtime.book !== book ||
          runtime.rendition !== rendition
        ) {
          locations.destroy();
          return;
        }
        const previousLocations = book.locations;
        book.locations = locations;
        previousLocations?.destroy();
        const hash=runtime.bookId?sourceDigests.get(runtime.bookId):null;
        if(hash&&isDesktop)void invoke('save_location_index',{sourceSha256:hash,locations:locations.save()}).catch(()=>{});
        void rendition.reportLocation();
      })
      .catch(() => {
        locations.destroy();
        const title = getBook(runtime.bookId)?.title ?? "当前书籍";
        showToast(`“${title}”阅读位置索引生成失败，仍可正常翻阅`, "error");
      }).finally(() => indexingBooks.delete(book));
  }, delay);
}

function updateGroupTail(rendition: Rendition, runtime: PaneRuntime): void {
  const internal = rendition as InternalRendition;
  const container = internal.manager.container;
  if (!container) return;
  if(runtime.readingMode==='scroll'){container.classList.remove('comfortable-group-tail');container.style.removeProperty('--comfortable-group-tail-width');return;}
  const lastView = internal.manager.views?.last();
  const atBookEnd = Boolean(lastView?.section && !lastView.section.next());
  const tailWidth = atBookEnd
    ? Math.max(0, runtime.actualPageCount - 1) * internal._layout.pageWidth
    : 0;
  container.classList.add("comfortable-group-tail");
  container.style.setProperty("--comfortable-group-tail-width", `${tailWidth}px`);
}

const originalLayouts=new WeakMap<Rendition,{calculate:InternalLayout['calculate'];count:InternalLayout['count']}>();
const sourceStrong=new WeakMap<EpubBook,Set<string>>();
function patchMultiPageLayout(rendition: Rendition, runtime: PaneRuntime): void {
  const internal = rendition as InternalRendition;
  const layout = internal._layout;
  if(!originalLayouts.has(rendition))originalLayouts.set(rendition,{calculate:layout.calculate.bind(layout),count:layout.count.bind(layout)});
  if(runtime.readingMode==='scroll'){const original=originalLayouts.get(rendition)!;layout.calculate=original.calculate;layout.count=original.count;internal.manager.updateLayout();return;}
  layout.calculate = (width: number, height: number, suppliedGap?: number) => {
    const divisor = Math.min(10, Math.max(1, runtime.actualPageCount));
    const pageWidth = width / divisor;
    const requestedGap = Number.isFinite(suppliedGap) ? Number(suppliedGap) : 18;
    const gap = divisor > 1
      ? Math.min(26, Math.max(0, Math.min(requestedGap, pageWidth - 12)))
      : Math.min(44,Math.max(24,width*.08));
    // epub.js adds half a gap at both viewport edges. Therefore every complete
    // visible page is exactly pageWidth = viewport / divisor, while its text
    // column is pageWidth - gap. No edge page can extend beyond the pane.
    const columnWidth = Math.max(12, pageWidth - gap);
    const spreadWidth = Math.max(12, width - gap);

    layout.width = width;
    layout.height = height;
    layout.spreadWidth = spreadWidth;
    layout.pageWidth = pageWidth;
    // One click advances one column, so 1–10 becomes a sliding window (1–4 → 2–5).
    layout.delta = pageWidth;
    layout.columnWidth = columnWidth;
    layout.gap = gap;
    layout.divisor = divisor;
    layout.update({
      width,
      height,
      spreadWidth,
      pageWidth,
      delta: pageWidth,
      columnWidth,
      gap,
      divisor,
    });
    // Add blank group slots only when the final spine section is mounted. A
    // permanent spacer behind an intermediate section would delay loading the
    // following chapter in the continuous manager.
    updateGroupTail(rendition, runtime);
  };
  layout.count = (totalLength: number, pageLength?: number) => {
    const unit = Math.max(1, pageLength || layout.pageWidth || layout.delta);
    const pages = Math.max(1, Math.ceil(totalLength / unit));
    return {
      spreads: Math.max(1, Math.ceil(pages / Math.max(1, runtime.actualPageCount))),
      pages,
    };
  };
  internal.manager.updateLayout();
}

/** Reflow mounted chapters in place. epub.js resize() clears every iframe first,
 * which flashes all books during a split adjustment and needlessly reloads assets.
 * Keep its layout/view sizing primitives, then restore our semantic anchor. */
function resizeMountedChapters(runtime:PaneRuntime):void {
  const rendition=runtime.rendition!;
  const manager=(rendition as any).manager;
  const views=manager?.views?.all?.()??[];
  if(!views.length||!manager.stage?.size||!manager.updateLayout){rendition.resize(runtime.layoutWidth,runtime.layoutHeight);return;}
  (rendition as any).settings.width=runtime.layoutWidth;
  (rendition as any).settings.height=runtime.layoutHeight;
  manager._stageSize=manager.stage.size(runtime.layoutWidth,runtime.layoutHeight);
  manager._bounds=manager.bounds();
  for(const view of views){
    view.settings.width=runtime.layoutWidth;view.settings.height=runtime.layoutHeight;
    view.size(runtime.layoutWidth,runtime.layoutHeight);
  }
  manager.updateLayout();
}

async function reflowPaneNow(index: number, force = false): Promise<void> {
  const runtime = runtimes[index];
  const rendition = runtime.rendition;
  const lazyPageBook = runtime.lazyPageCount > 0;
  const pane = paneElement(index);
  const stage = pane.querySelector<HTMLElement>(".pane-stage");
  const host = pane.querySelector<HTMLElement>(".epub-host");
  if ((!rendition && !lazyPageBook) || !stage || !host || stage.clientWidth < 50 || stage.clientHeight < 50) return;
  if (force && runtime.resizeActive) {
    if (runtime.resizeTimer !== null) window.clearTimeout(runtime.resizeTimer);
    if (runtime.resizeFrame !== null) window.cancelAnimationFrame(runtime.resizeFrame);
    runtime.resizeTimer = null;
    runtime.resizeFrame = null;
    runtime.resizeActive = false;
    runtime.resizeAnchorCfi = null;
    pane.classList.remove("live-resizing");
  }
  const available = availableStageSize(stage);
  const nextCount = runtime.readingMode==='scroll' ? 1 : runtime.pageMode > 0
    ? runtime.pageMode
    : automaticPageCountForWidth(available.width,index);
  const changed = nextCount !== runtime.actualPageCount;
  const viewportChanged =
    Math.abs(runtime.viewportWidth - available.width) > 1 ||
    Math.abs(runtime.viewportHeight - available.height) > 1;

  if (lazyPageBook) {
    if (runtime.pageMode > 0 && !changed && !force && runtime.layoutWidth >= 50) {
      if (viewportChanged) positionRenditionCanvas(index);
      return;
    }
    if (!changed && !force) return;
    runtime.actualPageCount = nextCount;
    if (runtime.pageMode === 0 || runtime.layoutWidth < 50 || runtime.layoutHeight < 50) {
      runtime.layoutWidth = available.width;
      runtime.layoutHeight = available.height;
    }
    positionRenditionCanvas(index);
    updatePaneHeader(index);
    await renderLazyPageWindow(index, runtime.currentPage || 1);
    return;
  }
  if (!rendition) return;

  // A pinned page mode owns a fixed logical canvas. Resizing, maximizing,
  // entering full screen, or changing the number of sibling panes only scales
  // that canvas. The page breaks and visible text therefore remain identical.
  if (runtime.readingMode!=='scroll' && runtime.pageMode > 0 && !changed && !force && runtime.layoutWidth >= 50) {
    if (viewportChanged) positionRenditionCanvas(index);
    return;
  }
  if (!changed && !viewportChanged && !force) {
    void rendition.reportLocation();
    return;
  }
  const anchor = runtime.readingMode==='scroll' ? runtime.resizeAnchorCfi??runtime.layoutAnchorCfi??runtime.cfi : runtime.layoutAnchorCfi??runtime.cfi;
  runtime.layoutAnchorCfi=anchor;
  runtime.preserveSemanticFocus=Boolean(anchor);
  runtime.actualPageCount = nextCount;
  if (force || runtime.readingMode==='scroll' || runtime.pageMode === 0 || runtime.layoutWidth < 50 || runtime.layoutHeight < 50) {
    runtime.layoutWidth = available.width;
    runtime.layoutHeight = available.height;
  }
  runtime.restoringLocation = true;
  positionRenditionCanvas(index);
  applyAdaptiveFontScale(index, host);
  updatePaneHeader(index);
  runtime.restoringLocation = true;
  try {
    resizeMountedChapters(runtime);
    if (anchor) await rendition.display(anchor);
    await settleRenditionLayout(index);
    fitAtomicBlocks(index);
    await settleRenditionLayout(index);
    if (anchor && runtime.rendition === rendition) {await rendition.display(anchor);await ensureAnchorVisible(index,anchor);}
  } finally {
    if (runtime.rendition === rendition) runtime.restoringLocation = false;
  }
  await settleBookLocation(index, rendition);
  scheduleAtomicFit(index);
  scheduleDynamicPageMap(index);
}

function setBookPageMode(index: number, requestedMode: number): void {
  const source = runtimes[index];
  if (!source.bookId) return;
  const mode = Math.min(10, Math.max(0, Math.round(requestedMode)));
  const existing = snapshot.progress[source.bookId] ?? {
    cfi: source.cfi,
    page: source.currentPage,
    totalPages: source.totalPages,
    percent: source.percent,
    updatedAt: Math.floor(Date.now() / 1000),
    pageMode: mode,
    annotations: [],
  };
  existing.pageMode = mode;
  snapshot.progress[source.bookId] = existing;

  runtimes.forEach((runtime, paneIndex) => {
    if (runtime.bookId !== source.bookId) return;
    runtime.pageMode = mode;
    updatePaneHeader(paneIndex);
    savePaneProgress(paneIndex, 0);
    void reflowPane(paneIndex, true);
  });
  showToast(
    mode === 0
      ? "已按阅读宽度自动分栏"
      : mode >= 6 ? `已切换到 ${mode} 栏概览` : `同屏显示 ${mode} 栏`,
  );
}


// A continuous rendition can report the previous section when only its empty
// column gutter intersects the viewport. Persist the first actual visible
// glyph/replaced element, so a chapter jump and later reopen share one anchor.
function firstVisibleContentCfi(index: number, slot = 0): string | null {
  const runtime = runtimes[index];
  const rendition = runtime.rendition;
  const stage = paneElement(index).querySelector<HTMLElement>(".pane-stage");
  if (!rendition || !stage) return null;
  const stageRect = stage.getBoundingClientRect();
  const scale = Math.max(0.001, runtime.viewportScale);
  const pageWidth = (rendition as InternalRendition)._layout.pageWidth * scale;
  const left = stageRect.left + 2 + Math.max(0, Math.min(runtime.actualPageCount - 1, slot)) * pageWidth;
  const right = Math.min(stageRect.right - 2, left + pageWidth);
  const top = stageRect.top + 2;
  const bottom = stageRect.bottom - 2;
  let best: { contents: EpubContents; range: Range; element?: Element; top: number; left: number } | null = null;
  for (const contents of visibleContents(rendition)) {
    const frame = contents.window.frameElement as HTMLIFrameElement | null;
    if (!frame || !frame.clientWidth) continue;
    const frameRect = frame.getBoundingClientRect();
    if (frameRect.right <= left || frameRect.left >= right) continue;
    const frameScale = frameRect.width / frame.clientWidth;
    const document = contents.document;
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT | NodeFilter.SHOW_ELEMENT);
    let node: Node | null;
    while ((node = walker.nextNode())) {
      const textNode = node.nodeType === Node.TEXT_NODE;
      if (textNode ? !node.textContent?.trim() : !(node as Element).matches("img,svg,video")) continue;
      const range = document.createRange();
      if (textNode) range.selectNodeContents(node); else range.selectNode(node);
      for (const rect of Array.from(range.getClientRects())) {
        const x = frameRect.left + rect.left * frameScale;
        const y = frameRect.top + rect.top * frameScale;
        const w = rect.width * frameScale;
        const h = rect.height * frameScale;
        if (w < 0.3 || h < 0.3 || x + w <= left + 0.5 || x >= right - 0.5 || y + h <= top || y >= bottom) continue;
        const visibleTop = Math.max(y, top);
        const visibleLeft = Math.max(x, left);
        if (best && (visibleTop > best.top + 1 || Math.abs(visibleTop - best.top) <= 1 && visibleLeft >= best.left)) continue;
        let anchor = range.cloneRange();
        if (textNode) {
          const localX = (visibleLeft - frameRect.left) / frameScale + 0.3;
          const localY = (visibleTop - frameRect.top) / frameScale + Math.min(rect.height / 2, 5);
          const caret = (document as Document & { caretRangeFromPoint?: (x: number, y: number) => Range | null }).caretRangeFromPoint?.(localX, localY);
          if (caret && caret.startContainer.nodeType === Node.TEXT_NODE) anchor = caret.cloneRange();
        }
        anchor.collapse(true);
        best = { contents, range: anchor, element: textNode ? undefined : node as Element, top: visibleTop, left: visibleLeft };
      }
    }
  }
  try { return best ? best.element ? best.contents.cfiFromNode(best.element, "reader-annotation") : best.contents.cfiFromRange(best.range, "reader-annotation") : null; }
  catch { return null; }
}

function currentSectionHref(index: number): string | null {
  const runtime = runtimes[index];
  if (!runtime.book || !runtime.rendition) return null;
  const contents = visibleContents(runtime.rendition)[0];
  return runtime.book.spine.get(contents?.sectionIndex ?? runtime.cfi ?? 0)?.href ?? null;
}

function updateChapterBoundary(index: number): void {
  const pane = paneElement(index);
  const boundary = pane.querySelector<HTMLElement>(".pane-chapter-boundary");
  const previous = boundary?.querySelector<HTMLButtonElement>(".chapter-boundary-prev");
  const next = boundary?.querySelector<HTMLButtonElement>(".chapter-boundary-next");
  const runtime = runtimes[index];
  const session = runtime.bookId ? portableBooks.get(runtime.bookId) : null;
  const chapters = Array.isArray(session?.book?.chapters) ? session.book.chapters : [];
  const section = currentSectionHref(index);
  const container = runtime.rendition ? (runtime.rendition as InternalRendition).manager.container : null;
  if (!boundary || !previous || !next || !runtime.streamed || runtime.readingMode !== "scroll" || !section || !container) {
    if (boundary) boundary.hidden = true;
    if (previous) previous.hidden = true;
    if (next) next.hidden = true;
    return;
  }
  const filename = (path: string): string => path.split("#", 1)[0].split("?")[0].split("/").pop() ?? "";
  const chapterIndex = chapters.findIndex((chapter: any) => filename(chapter.readingDocument?.path ?? "") === filename(section));
  const atStart = container.scrollTop <= 2;
  const atEnd = container.scrollHeight - container.clientHeight - container.scrollTop <= 2;
  const previousChapter = atStart && chapterIndex > 0 ? chapters[chapterIndex - 1] : null;
  const nextChapter = atEnd && chapterIndex >= 0 && chapterIndex + 1 < chapters.length ? chapters[chapterIndex + 1] : null;
  for (const [button, chapter] of [[previous, previousChapter], [next, nextChapter]] as const) {
    button.hidden = !chapter;
    if (!chapter) continue;
    button.dataset.sourceHref = section;
    button.dataset.targetHref = filename(chapter.readingDocument.path);
    button.setAttribute("aria-label", `${button === previous ? "上一章" : "下一章"}：${chapter.title ?? "继续阅读"}`);
  }
  boundary.hidden = !previousChapter && !nextChapter;
}

function openBoundaryChapter(index: number, button: HTMLButtonElement): void {
  const runtime = runtimes[index];
  const bookId = runtime.bookId, generation = runtime.generation, rendition = runtime.rendition;
  const source = button.dataset.sourceHref, target = button.dataset.targetHref;
  if (!runtime.streamed || runtime.readingMode !== "scroll" || !bookId || !rendition || !source || !target || currentSectionHref(index) !== source) return;
  void enqueuePaneOperation(index, async () => {
    const active = runtimes[index];
    if (active.generation !== generation || active.bookId !== bookId || active.rendition !== rendition || currentSectionHref(index) !== source) return;
    await performBookJump(index, target, false, true);
    if (runtimes[index].generation === generation && runtimes[index].bookId === bookId) updateChapterBoundary(index);
  });
}

function contentCfiVisible(index: number, cfi: string | null): boolean {
  const rendition = runtimes[index].rendition;
  if (!rendition || !cfi?.startsWith("epubcfi(")) return false;
  try {
    const range = rendition.getRange(cfi, "reader-annotation");
    if (!range) return false;
    const doc = range.startContainer.ownerDocument;
    const frame = Array.from(paneElement(index).querySelectorAll<HTMLIFrameElement>(".epub-host iframe"))
      .find(candidate => candidate.contentDocument === doc);
    const viewport = paneElement(index).querySelector<HTMLElement>(".pane-stage")?.getBoundingClientRect();
    if (!frame || !viewport || !frame.clientWidth) return false;
    const frameBox = frame.getBoundingClientRect();
    const scale = frameBox.width / frame.clientWidth;
    const intersects = (rect: DOMRect): boolean => {
      if (!Number.isFinite(scale) || scale <= 0 || rect.height <= 0) return false;
      const x = frameBox.left + rect.left * scale;
      const y = frameBox.top + rect.top * scale;
      return x < viewport.right - 2 && x + Math.max(1, rect.width * scale) > viewport.left + 2 &&
        y < viewport.bottom - 2 && y + rect.height * scale > viewport.top + 2;
    };
    if (Array.from(range.getClientRects()).some(intersects)) return true;
    // A CFI at a paragraph's first character may sit just above the viewport
    // while the rest of that same short paragraph is still being read.
    const start = range.startContainer.nodeType === Node.ELEMENT_NODE
      ? range.startContainer as Element : range.startContainer.parentElement;
    const block = start?.closest("p,li,blockquote,figcaption,h1,h2,h3,h4,h5,h6");
    return Boolean(block && Array.from(block.getClientRects()).some(rect => rect.height <= viewport.height * 1.5 && intersects(rect)));
  } catch { return false; }
}

const singleSectionPageCounts=new WeakMap<Document,{key:string;count:number}>();
function measuredSingleSectionPages(runtime:PaneRuntime):number|null{
  if(!runtime.rendition||!runtime.book||(runtime.book.spine as InternalSpine).spineItems.length!==1)return null;
  const contents=visibleContents(runtime.rendition)[0];const doc=contents?.document;if(!doc?.body||doc.fonts?.status==="loading")return null;
  const layout=(runtime.rendition as InternalRendition)._layout;const pageWidth=layout.pageWidth;if(!(pageWidth>1))return null;
  const key=`${runtime.layoutWidth}|${runtime.layoutHeight}|${runtime.actualPageCount}|${runtime.effectiveFontScale}|${readingPreferences(runtimes.indexOf(runtime)).readerFont}`;
  const previous=singleSectionPageCounts.get(doc);if(previous?.key===key)return previous.count;
  let right=0;const walker=doc.createTreeWalker(doc.body,NodeFilter.SHOW_TEXT);let node:Node|null;
  while((node=walker.nextNode())){if(!node.textContent?.trim()||node.parentElement?.closest('script,style,annotation'))continue;const range=doc.createRange();range.selectNodeContents(node);for(const rect of Array.from(range.getClientRects()))if(rect.width>0&&rect.height>0)right=Math.max(right,rect.right);}
  for(const object of Array.from(doc.querySelectorAll<HTMLElement>('img,svg,video,table'))){for(const rect of Array.from(object.getClientRects()))if(rect.width>0&&rect.height>0)right=Math.max(right,rect.right);}
  if(!right)return null;const count=Math.max(1,Math.ceil((right-1)/pageWidth));singleSectionPageCounts.set(doc,{key,count});return count;
}

function handleRelocated(index: number, location: EpubLocation): void {
  const runtime = runtimes[index];
  if (!runtime.book || !runtime.bookId) return;
  if (runtime.rendition) updateGroupTail(runtime.rendition, runtime);
  let cfi = location.start?.cfi;
  if (!cfi) return;
  // epub.js can briefly report its default first page while a saved CFI is
  // still being restored. Never let that transient event overwrite progress.
  if (runtime.restoringLocation || runtime.resizeActive) return;
  const focus = runtime.preserveSemanticFocus ? runtime.layoutAnchorCfi : null;
  if (focus && contentCfiVisible(index, focus)) cfi = focus;
  else {
    runtime.preserveSemanticFocus = false;
    if (linearSectionCount(runtime.book) > 1) cfi = firstVisibleContentCfi(index) ?? cfi;
    runtime.layoutAnchorCfi = cfi;
  }
  const fallbackPage = location.start.displayed?.page ?? 1;
  const fallbackTotal = location.start.displayed?.total ?? 1;
  if (runtime.readingMode!=='scroll' && linearSectionCount(runtime.book) === 1) {
    runtime.totalPages = measuredSingleSectionPages(runtime) ?? Math.max(1, fallbackTotal);
    const internal = runtime.rendition as InternalRendition | null;
    const container = internal?.manager.container;
    // Chromium quantizes fractional scroll positions. epub.js floors them,
    // which can label column 3 as page 2 at widths not divisible by three.
    const exactSingleView = (runtime.book.spine as InternalSpine).spineItems.length === 1;
    const snappedPage = exactSingleView && container && internal && internal._layout.settings.direction !== "rtl"
      ? Math.round(container.scrollLeft / internal._layout.pageWidth) + 1 : fallbackPage;
    runtime.currentPage = Math.min(runtime.totalPages, Math.max(1, snappedPage));
    runtime.endPage = Math.min(
      runtime.totalPages,
      runtime.currentPage + Math.max(1, runtime.actualPageCount) - 1,
    );
    runtime.percent = runtime.totalPages > 1
      ? Math.min(1, Math.max(0, (runtime.currentPage - 1) / (runtime.totalPages - 1)))
      : 0;
    runtime.cfi = cfi;
    updatePaneProgress(index);
    applyTextAnnotations(index);
    renderOverlayAnnotations(index);
    savePaneProgress(index);
    updateChapterBoundary(index);if(index===snapshot.session.activePane)markCurrentToc();
    return;
  }
  const total = runtime.book.locations.length();
  const locatedResult = total > 0 ? runtime.book.locations.locationFromCfi(cfi) : -1;
  const located = typeof locatedResult === "number" ? locatedResult : -1;
  runtime.currentPage = located >= 0 ? located + 1 : fallbackPage;
  // The visible range is defined by the reader's complete-page window, not by
  // the EPUB location chunk nearest the right edge (chunks are only a page-map
  // approximation and can otherwise under-count a visible page).
  runtime.endPage = Math.min(
    total || fallbackTotal,
    runtime.currentPage + Math.max(1, runtime.actualPageCount) - 1,
  );
  runtime.totalPages = total > 0 ? total : fallbackTotal;
  runtime.percent =
    total > 1
      ? Math.min(1, Math.max(0, located / (total - 1)))
      : Math.min(1, Math.max(0, location.start.percentage ?? 0));
  runtime.cfi = cfi;
  updatePaneProgress(index);
  applyTextAnnotations(index);
  renderOverlayAnnotations(index);
  savePaneProgress(index);
  updateChapterBoundary(index);if(index===snapshot.session.activePane)markCurrentToc();
}

async function openBook(bookId: string, index = snapshot.session.activePane,requestedChapter?:string,activate=true): Promise<void> {
  const previousActive=snapshot.session.activePane;
  const alreadyOpen=snapshot.session.paneBookIds.findIndex((id,i)=>id===bookId&&i!==index);
  if(alreadyOpen>=0){setActivePane(alreadyOpen);if(requestedChapter)await openBook(bookId,alreadyOpen,requestedChapter,activate);return;}
  const openingRequest=++bookOpenSequence[index];
  if(learningIsOpen()){await closeLearning();if(learningIsOpen())return;}
  if(openingRequest!==bookOpenSequence[index])return;
  const bookRecord = getBook(bookId);
  if (!bookRecord) {
    showToast("这本书已不在本地书库中", "error");
    return;
  }
  if(activate)setActivePane(index);
  const runtime = runtimes[index];
  const sourceKey=JSON.stringify([bookRecord.path,bookRecord.modifiedAt,bookRecord.catalogSource?.sha256]);
  if (runtime.bookId === bookId && runtime.openedSourceKey===sourceKey&&(runtime.rendition || runtime.lazyPageCount > 0)) {
    if(requestedChapter){const chapter=portableBooks.get(bookId)?.book.chapters.find((item:any)=>item.id===requestedChapter);if(!chapter)throw new Error('当前内容版本没有这个章节，原书页保持打开');await jumpToBookTarget(chapter.readingDocument.path.split('/').pop(),false,index);}
    return;
  }

  try{const previous=makePaneProgress(index);if(previous)await writePaneProgress(index,previous);else await progressWrites[index];}catch(error){showToast('原位置尚未保存，书页保持打开：'+String(error),'error');return;}
  if(openingRequest!==bookOpenSequence[index])return;
  destroyRuntime(index, false, false);
  runtime.bookId = bookId;
  runtime.openedSourceKey=sourceKey;
  snapshot.session.paneBookIds[index] = bookId;
  snapshot.session.paneCount=Math.max(snapshot.session.paneCount,index+1);
  if(workspace&&!workspace.ids.includes(index))workspace.added(index,previousActive);
  updatePaneHeader(index);
  updateActiveUi();
  persistSession();

  const pane = paneElement(index);
  const stage = pane.querySelector<HTMLElement>(".pane-stage");
  const host = pane.querySelector<HTMLElement>(".epub-host");
  if (!stage || !host) return;
  pane.classList.add("loading");
  const generation = runtime.generation;
  let finishOpening!:()=>void;
  const opening=new Promise<void>(resolve=>{finishOpening=resolve;});
  runtime.opening=opening;runtime.finishOpening=finishOpening;

  try {
    if (bookRecord.lazyPages && bookRecord.lazyPages > 0) {
      await openLazyPageBook(
        index,
        generation,
        bookRecord.lazyPages,
        snapshot.progress[bookId],
        stage,
        host,
      );
      return;
    }
    let portable=null;
    if(bookRecord.catalogSource){try{portable=await preparePortableBook(bookRecord as CatalogRecord);}catch(error){portableBooks.delete(bookId);if(!isDesktop||/^https?:\/\//.test(bookRecord.path))throw error;showToast('在线材料暂未取得，先读取本机正文。');}}
    const streamed=Boolean(portable&&/^https?:\/\//.test(bookRecord.path));
    runtime.streamed=streamed;
    void prepareLearning(bookId);
    const raw = streamed?null:await invoke<ArrayBuffer | number[]>("load_book_bytes", { bookId });
    if (generation !== runtime.generation) return;
    const bytes=raw?rawResponseToBuffer(raw):null;
    const sourceHash=streamed?(portable!.book.reader.epubSha256??portable!.record.catalogSource.sha256):Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256",bytes!))).map(x=>x.toString(16).padStart(2,"0")).join("");
    if(generation!==runtime.generation)return;
    sourceDigests.set(bookId,sourceHash);
    const contentDigest=streamed?portable!.book.reader.contentDigest:await invoke<string>('epub_content_identity',{bookId,sourceSha256:sourceHash}).catch(()=>null);
    if(generation!==runtime.generation)return;
    if(contentDigest)contentDigests.set(bookId,contentDigest);
    const priorHash=snapshot.progress[bookId]?.sourceSha256;
    const priorContent=snapshot.progress[bookId]?.contentDigest;
    const changed=priorContent&&contentDigest?priorContent!==contentDigest:priorHash&&(streamed?priorHash!==portable!.record.catalogSource.sha256:priorHash!==sourceHash);
    if(changed){
      try{snapshot.progress[bookId]=await invoke<BookProgress>('activate_progress_edition',{bookId,sourceSha256:sourceHash,contentDigest});sourceMismatches.delete(bookId);showToast(snapshot.progress[bookId].cfi?'已恢复这份内容版本自己的位置与批注。':'已打开新的内容版本。原版位置与批注保留在“版本记录”，没有直接贴到新正文。');}
      catch(error){sourceMismatches.add(bookId);showToast('旧记录保持完整，但新版本记录尚未建立：'+String(error),'error');}
    }
    else{sourceMismatches.delete(bookId);if(contentDigest)ensureBookProgress(bookId,index).contentDigest=contentDigest;ensureBookProgress(bookId,index).bookUuid=bookRecord.bookUuid;}
    if(portable&&!streamed&&contentDigest&&contentDigest!==portable.book.reader.contentDigest){portableBooks.delete(bookId);portable=null;holdLearning(bookId,'学习材料与本机正文的版本不同；普通阅读与原记录保留，请核对后重新连接材料。');}
    const book = streamed?ePub(readerPackageUrl(bookId),{requestMethod:portableRequest(portable!),replacements:'none'}):ePub(bytes!);
    propagateBookOpenFailure(book as unknown as Parameters<typeof propagateBookOpenFailure>[0]);
    runtime.book = book;
    await Promise.all([book.ready, book.opened]);
    try{const archive=(book as any).archive?.zip;const name=Object.keys(archive?.files??{}).find(name=>/\/original\/source\.md$/i.test(name));if(name){const file=archive.file(name);if((file?._data?.uncompressedSize??0)<=4*1024*1024)sourceStrong.set(book,sourceStrongPatterns(await file.async('string')));}}catch{/* No compatible original: preserve the publisher's content as it is. */}
    if(portable)book.locations.load(JSON.stringify(portable.book.reader.locations));
    else if(isDesktop){try{const cached=await invoke<string|null>('load_location_index',{sourceSha256:sourceHash});if(cached)book.locations.load(cached);}catch{/* Reading remains usable without an optional index cache. */}}
    refreshLearningButton();
    if (generation !== runtime.generation) {
      book.destroy();
      return;
    }

    // Window-state restoration and an eager user drag can otherwise race the
    // first CFI display. Capture the fixed logical canvas only after the native
    // window has been quiet for a short interval.
    await waitForStableStage(stage);

    const saved = snapshot.progress[bookId];
    runtime.readingMode=saved?.readingMode==='scroll'?'scroll':saved?.readingMode==='paged'||isDesktop?'paged':'scroll';
    runtime.pageMode = pageModeForBook(bookId);
    const available = availableStageSize(stage);
    runtime.actualPageCount = runtime.readingMode==='scroll' ? 1 : runtime.pageMode > 0
      ? runtime.pageMode
      : automaticPageCountForWidth(available.width,index);
    runtime.layoutWidth = available.width;
    runtime.layoutHeight = available.height;
    positionRenditionCanvas(index);
    updatePaneHeader(index);

    const rendition = new Rendition(book, {
      // Numeric dimensions prevent epub.js Stage from registering its own
      // throttled window.resize handler. The app is the sole resize owner;
      // otherwise epub.js clears every view repeatedly while a border is dragged.
      width: runtime.layoutWidth,
      height: runtime.layoutHeight,
      // Continuous keeps adjacent EPUB spine sections mounted, so a one-page
      // slide remains continuous even when the visible window crosses chapters.
      manager: "continuous",
      ...(streamed?{offset:0,offsetDelta:0}:{}),
      flow: runtime.readingMode==='scroll'?'scrolled-continuous':"paginated",
      spread: "none",
      infinite: true,
      snap: false,
      allowScriptedContent: false,
      ignoreClass: "reader-annotation",
      // The application owns resizing. In a fixed page mode the EPUB canvas is
      // intentionally not reflowed when the outer window changes size.
      resizeOnOrientationChange: false,
    });
    runtime.rendition = rendition;
    propagateDisplayFailure(rendition as unknown as Parameters<typeof propagateDisplayFailure>[0]);
    await rendition.started;
    if (generation !== runtime.generation) {
      rendition.destroy();
      return;
    }
    patchMultiPageLayout(rendition, runtime);
    await rendition.attachTo(host);
    const markUserScroll=()=>{if(runtime.rendition===rendition&&runtime.readingMode==='scroll')runtime.preserveSemanticFocus=false;};
    (rendition as InternalRendition).manager.container?.addEventListener('wheel',markUserScroll,{passive:true});
    (rendition as InternalRendition).manager.container?.addEventListener('touchstart',markUserScroll,{passive:true});
    (rendition as InternalRendition).manager.container?.addEventListener('pointerdown',markUserScroll,{passive:true});
    let overlayFrame=0;
    (rendition as InternalRendition).manager.container?.addEventListener('scroll',()=>{if(runtime.readingMode==='scroll'&&!overlayFrame)overlayFrame=requestAnimationFrame(()=>{overlayFrame=0;if(runtime.rendition===rendition){renderOverlayAnnotations(index);updateChapterBoundary(index);}});},{passive:true});
    rendition.hooks.content.register((contents: EpubContents) => applyReadingTheme(contents));
    applyAdaptiveFontScale(index, host);
    updatePaneHeader(index);
    book.spine.hooks.serialize.register(async(output: string, section: { output: string; url:string }) => {
      if(streamed)section.output=await rewritePortableResources(bookId,section.output||output,section.url);
      prepareSectionForLayout(index,section.output||output,section);
    });
    const requestedHref=requestedChapter&&portable?portable.book.chapters.find((c:any)=>c.id===requestedChapter)?.readingDocument.path.split('/').pop():null;
    const restoreCfi = requestedHref??(editionHeld(bookId)?null:saved?.cfi ?? null);
    runtime.layoutAnchorCfi=restoreCfi;
    runtime.preserveSemanticFocus=Boolean(restoreCfi);
    runtime.restoringLocation = Boolean(restoreCfi);
    rendition.hooks.content.register((contents: EpubContents) => wireBookDocument(index, contents));
    rendition.on("relocated", (location: EpubLocation) => handleRelocated(index, location));
    rendition.on("selected", (cfiRange: string, contents: EpubContents) =>
      handleTextSelection(index, cfiRange, contents),
    );
    rendition.on("rendered", () => {
      if (!runtime.restoringLocation) pane.classList.remove("loading");
      updateGroupTail(rendition, runtime);
      scheduleAtomicFit(index);
      window.requestAnimationFrame(() => {
        applyTextAnnotations(index);
        renderOverlayAnnotations(index);
      });
    });

    try {
      await rendition.display(restoreCfi ?? undefined);
      // A native resize during startup can race the first display request.
      // Reassert the persisted anchor once before exposing/saving the page.
      await settleRenditionLayout(index);
      fitAtomicBlocks(index);
      await settleRenditionLayout(index);
      if (restoreCfi) {await rendition.display(restoreCfi);await ensureAnchorVisible(index,restoreCfi);}
    } catch (error) {
      if (error instanceof ReaderContentLoadError) throw error;
      if (!restoreCfi) throw error;
      await rendition.display();
      showToast("原阅读位置已失效，已从书籍开头打开", "error");
    } finally {
      runtime.restoringLocation = false;
    }
    await rendition.reportLocation();
    updateChapterBoundary(index);
    if (index === snapshot.session.activePane) renderChapterList();
    pane.classList.remove("loading");
    applyTextAnnotations(index);
    renderOverlayAnnotations(index);

    runtime.resizeObserver = new ResizeObserver((entries) => {
      if (!runtime.rendition) return;
      const rect = entries[0]?.contentRect;
      scheduleLiveViewportScale(
        index,
        rect?.width ?? stage.clientWidth,
        rect?.height ?? stage.clientHeight,
      );
    });
    runtime.resizeObserver.observe(stage);

    scheduleAtomicFit(index);
    scheduleDynamicPageMap(index, 0);
    savePaneProgress(index);
  } catch (error) {
    if(generation!==runtime.generation)return;
    pane.classList.remove("loading");
    destroyRuntime(index, true);
    runtime.bookId = bookId;
    pane.classList.add("load-error");
    showToast(`无法打开“${bookRecord.title}”：${readableError(error)}`, "error");
  } finally {
    finishOpening();
    if(runtime.opening===opening){runtime.opening=null;runtime.finishOpening=null;}
  }
}

function setActivePane(index: number): void {
  if (index < 0 || index >= MAX_PANES) return;
  if(snapshot.session.activePane===index)return;
  snapshot.session.activePane = index;
  updateActiveUi();
  renderChapterList();
  updateNavigationHistoryUi();
  persistSession();
}

async function ensureLocationsForJump(index: number): Promise<boolean> {
  const runtime = runtimes[index];
  if (!runtime.book || !runtime.rendition) return false;
  if (runtime.book.locations.length() > 0) return true;
  scheduleDynamicPageMap(index, 0, true);
  for (let attempt = 0; attempt < 80; attempt += 1) {
    if (runtime.book.locations.length() > 0) return true;
    await new Promise<void>((resolve) => window.setTimeout(resolve, 100));
  }
  return runtime.book.locations.length() > 0;
}

async function jumpToPage(index: number, requestedPage: number): Promise<void> {
  const runtime = runtimes[index];
  if (!runtime.bookId) return;
  setActivePane(index);
  if (runtime.lazyPageCount > 0) {
    await renderLazyPageWindow(index, requestedPage);
    return;
  }
  if (!runtime.rendition || !runtime.book) return;
  if (runtime.readingMode!=='scroll' && linearSectionCount(runtime.book) === 1) { await jumpToScreenPage(index, requestedPage); return; }
  const ready = await ensureLocationsForJump(index);
  if (!ready) {
    showToast("阅读位置索引仍在生成，请稍后再试", "error");
    return;
  }
  const total = runtime.book.locations.length();
  const bodyTarget = learningCfiFromBodyPosition(runtime.bookId, runtime.book, runtime.cfi, requestedPage);
  if (bodyTarget) { await jumpToBookTarget(bodyTarget, false, index); return; }
  const page = Math.min(Math.max(1, Math.round(requestedPage)), Math.max(1, total));
  const percentage = total > 1 ? (page - 1) / (total - 1) : 0;
  const cfi = runtime.book.locations.cfiFromPercentage(percentage);
  await jumpToBookTarget(cfi, false, index);
}

type NavigationDistance = "single" | "group";

async function navigateNow(
  index: number,
  direction: "prev" | "next",
  distance: NavigationDistance = "single",
): Promise<void> {
  const runtime = runtimes[index];
  if (runtime?.lazyPageCount > 0) {
    const step = distance === "group" ? Math.max(1, runtime.actualPageCount) : 1;
    const current = Math.max(1, runtime.currentPage);
    const target = distance === "group"
      ? (() => {
          const groupStart = Math.floor((current - 1) / step) * step + 1;
          const lastGroupStart = Math.floor((runtime.lazyPageCount - 1) / step) * step + 1;
          return direction === "next"
            ? Math.min(lastGroupStart, groupStart + step)
            : Math.max(1, groupStart - step);
        })()
      : direction === "next"
        ? Math.min(runtime.lazyPageCount, current + 1)
        : Math.max(1, current - 1);
    await renderLazyPageWindow(index, target);
    return;
  }
  const rendition = runtime?.rendition;
  if (!rendition) {
    openDrawer();
    return;
  }
  if(runtime.readingMode==='scroll'){
    runtime.preserveSemanticFocus=false;
    const container=(rendition as InternalRendition).manager.container;
    if(container){container.scrollTop+=(direction==='next'?1:-1)*(distance==='group'?container.clientHeight*.88:80);await new Promise<void>(resolve=>requestAnimationFrame(()=>resolve()));await rendition.reportLocation();updateChapterBoundary(index);}
    return;
  }
  if(runtime.book&&(runtime.book.spine as InternalSpine).spineItems.length===1){
    if(direction==="prev"&&runtime.currentPage<=1)return;
    if(direction==="next"&&(distance==="group"?runtime.endPage:runtime.currentPage)>=runtime.totalPages)return;
  }
  runtime.preserveSemanticFocus=false;
  const internal = rendition as InternalRendition;
  const layout = internal._layout;
  const originalDelta = layout.delta;
  const originalPropsDelta = layout.props.delta;
  let navigated = false;
  try {
    if(runtime.streamed){
      const manager=internal.manager as any;
      if(manager.check&&manager.settings){const offset=manager.settings.offset;runtime.restoringLocation=true;try{manager.settings.offset=originalDelta*(distance==='group'?Math.max(1,runtime.actualPageCount):1);await manager.check();await settleRenditionLayout(index);}finally{manager.settings.offset=offset;runtime.restoringLocation=false;}}
    }
    if (distance === "group") {
      const groupDelta = originalDelta * Math.max(1, runtime.actualPageCount);
      layout.delta = groupDelta;
      layout.props.delta = groupDelta;
    }
    await rendition[direction]();
    navigated = true;
  } catch (error) {
    showToast(`翻页失败：${String(error)}`, "error");
  } finally {
    layout.delta = originalDelta;
    if (originalPropsDelta === undefined) delete layout.props.delta;
    else layout.props.delta = originalPropsDelta;
  }
  if (navigated && distance === "group") {
    updateGroupTail(rendition, runtime);
    await rendition.reportLocation();
  }
}

function navigateActive(
  direction: "prev" | "next",
  distance: NavigationDistance = "single",
): void {
  void navigate(snapshot.session.activePane, direction, distance);
}

async function closeBook(index: number): Promise<void> {
  const request=++bookOpenSequence[index];
  try{const previous=makePaneProgress(index);if(previous)await writePaneProgress(index,previous);else await progressWrites[index];}catch(error){showToast('原位置尚未保存，书页保持打开：'+String(error),'error');return;}
  if(request!==bookOpenSequence[index])return;
  destroyRuntime(index, false, false);
  snapshot.session.paneBookIds[index] = null;
  workspace.removed(index);
  if(snapshot.session.activePane===index)snapshot.session.activePane=workspace.active;
  updatePaneHeader(index);
  updatePaneProgress(index);
  updateActiveUi();
  persistSession();
}

async function openBookInNewPane(bookId: string,requestedChapter?:string): Promise<void> {
  const alreadyOpen = snapshot.session.paneBookIds.findIndex((id) => id === bookId);
  if (alreadyOpen >= 0) {
    setActivePane(alreadyOpen);
    if(requestedChapter)await openBook(bookId,alreadyOpen,requestedChapter);
    return;
  }
  const target = snapshot.session.paneBookIds.findIndex((id) => !id);
  if (target < 0) {
    showToast(`已经打开 ${MAX_PANES} 本书。请先关闭一个阅读区域，书和位置会保留。`);return;
  }
  await openBook(bookId, target,requestedChapter);
}

function renderLibraryAudit(): void {
  if(!isDesktop){libraryAudit.innerHTML='<strong>我的书籍</strong><span>这里保留读过的书；完整材料只在主动选择时保存。</span>';return;}
  const scan = snapshot.scan;
  const hasWarning =
    scan.unreadable > 0 || scan.missingRoots > 0 || scan.otherBookFiles > 0 || snapshot.books.some(book=>book.available===false);
  const allCandidatesLoaded = scan.loadedCandidates === scan.epubCandidates;
  libraryAudit.classList.toggle("warning", hasWarning || !allCandidatesLoaded);

  const details = [
    `递归扫描 ${scan.rootsScanned} 个书库`,
    `EPUB ${scan.loadedCandidates}/${scan.epubCandidates}`,
  ];
  if (scan.duplicates) details.push(`去重 ${scan.duplicates}`);
  if (scan.ignoredTrees) details.push(`排除回收站 ${scan.ignoredTrees}`);
  if (scan.otherBookFiles) details.push(`待转换 ${scan.otherBookFiles}`);
  if (scan.missingRoots) details.push(`离线目录 ${scan.missingRoots}`);
  if (scan.unreadable) details.push(`不可读 ${scan.unreadable}`);
  const offline=snapshot.books.filter(book=>book.available===false).length;if(offline)details.push(`暂不可用的已登记文件 ${offline}`);

  libraryAudit.innerHTML = `
    <strong>${
      !hasWarning && allCandidatesLoaded
        ? `书目已更新 · ${scan.booksLoaded} 本书`
        : `已读取 ${scan.booksLoaded} 本，另有项目需处理`
    }</strong>
    <span>${details.join(" · ")}</span>`;
  libraryAudit.title = [
    `共享书库清单：${snapshot.libraryRoots.join("；") || "尚未登记"}`,
    ...scan.issues.map((issue) => `${issue.path}：${issue.message}`),
  ].join("\n");
}

function renderLibrary(query = ""): void {
  renderLibraryAudit();
  const normalizedQuery = query.trim().toLocaleLowerCase("zh-CN");
  const books = snapshot.books.filter((book) =>
    `${book.title} ${book.author}`.toLocaleLowerCase("zh-CN").includes(normalizedQuery),
  );
  const count = app.querySelector<HTMLElement>(".book-count");
  if (count) count.textContent = String(snapshot.books.length);
  if (books.length === 0) {
    libraryList.innerHTML = `
      <div class="empty-library">
        ${icons.book}
        <strong>${snapshot.books.length ? "没有匹配的书" : isDesktop ? "书库还是空的" : "还没有读过的书"}</strong>
        <span>${snapshot.books.length ? "换个关键词试试" : isDesktop ? "添加 EPUB，或选择一个已有书库文件夹" : "从书目中发现一本书，开始阅读"}</span>
      </div>`;
    return;
  }
  libraryList.innerHTML = books
    .map((book, index) => {
      const progress = snapshot.progress[book.id];
      const metric = progress ? readingMetricLabel(progress.readingMetric) : {text: '尚未阅读', percent: null};
      const hue = (index * 43 + book.title.length * 17) % 360;
      const isOpen = snapshot.session.paneBookIds.includes(book.id);
      return `
        <article class="library-book ${isOpen ? "open" : ""}" role="listitem" data-book-id="${escapeHtml(book.id)}">
          <button class="book-main" type="button" title="${isOpen?'回到这本书':'打开并加入阅读空间'}">
            <span class="mini-cover" style="--cover-hue:${hue}">${escapeHtml(book.title.slice(0, 1))}</span>
            <span class="book-copy">
              <strong>${escapeHtml(book.title)}</strong>
              <em>${escapeHtml(book.author)}</em>
              ${metric.percent === null ? '' : `<span class="book-progress"><i style="width:${metric.percent}%"></i></span>`}
              <small>${escapeHtml(metric.text)}</small>
            </span>
          </button>
          <button class="book-new-pane" type="button" title="${isOpen?'回到这本书':'放在旁边对照'}" aria-label="${isOpen?'回到这本书':'放在旁边对照'}">${
            isOpen ? icons.check : icons.plus
          }</button>
        </article>`;
    })
    .join("");
}

async function importSelectedBooks(): Promise<void> {
  const selected = await open({
    multiple: true,
    directory: false,
    title: "选择要加入书库的 EPUB",
    filters: [{ name: "EPUB 电子书", extensions: ["epub"] }],
  });
  if (!selected) return;
  const paths = Array.isArray(selected) ? selected : [selected];
  await importPaths(paths);
}

async function importFolder(): Promise<void> {
  const selected = await open({
    multiple: false,
    directory: true,
    title: "选择本地书库文件夹",
  });
  if (!selected) return;
  await importPaths([selected]);
}

async function importPaths(paths: string[]): Promise<void> {
  try {
    const next = await invoke<AppSnapshot>("import_paths", { paths });
    snapshot = {
      ...next,
      progress: normalizeProgress(next.progress),
      session: normalizeSession(next.session),
    };
    renderLibrary(searchInput.value);
    const failed = snapshot.scan.unreadable + snapshot.scan.missingRoots;
    showToast(
      failed
        ? `书库共 ${snapshot.books.length} 本；有 ${failed} 项未读取，请查看书库核对栏`
        : `书库已完整更新，共 ${snapshot.books.length} 本书`,
      failed ? "error" : "normal",
    );
  } catch (error) {
    showToast(String(error), "error");
  }
}

async function refreshLibrary(): Promise<void> {
  try {
    const next = await invoke<AppSnapshot>("refresh_library");
    snapshot.books = next.books;
    snapshot.progress = normalizeProgress(next.progress);
    snapshot.libraryRoots = next.libraryRoots;
    snapshot.scan = next.scan;
    renderLibrary(searchInput.value);
    const failed = snapshot.scan.unreadable + snapshot.scan.missingRoots;
    showToast(
      failed
        ? `已读取 ${snapshot.books.length} 本；有 ${failed} 项未读取，请查看书库核对栏`
        : `书库已更新，共 ${snapshot.books.length} 本书`,
      failed ? "error" : "normal",
    );
  } catch (error) {
    showToast(String(error), "error");
  }
}

function cycleTheme(): void {
  const themes: ThemeName[] = ["paper", "light", "night", "contrast"];
  const current = themes.indexOf(snapshot.session.theme);
  snapshot.session.theme = themes[(current + 1) % themes.length];
  applyTheme();
  persistSession();
}

function changeFont(delta: number): void {
  changeReadingPreference('fontScale',Math.min(180,Math.max(70,readingPreferences(snapshot.session.activePane).fontScale+delta)));
}

async function toggleFullscreen(): Promise<void> {
  const appWindow = getCurrentWindow();
  const next = !(await appWindow.isFullscreen());
  await appWindow.setFullscreen(next);
  showToast(next ? "已进入全屏，按 F11 退出" : "已退出全屏");
}

function wireEvents(): void {
  app.querySelector('.tools-toggle')?.addEventListener('click',()=>setToolsVisible(!toolsVisible));
  app.querySelector('.tools-close')?.addEventListener('click',()=>setToolsVisible(false));
  app.querySelector('.workspace-library')?.addEventListener('click',()=>drawer.classList.contains('visible')?closeDrawer():openDrawer());
  window.addEventListener('reader-learning-ready',event=>{const id=(event as CustomEvent).detail.bookId;for(const [index,runtime] of runtimes.entries()){if(runtime.bookId!==id||!runtime.rendition)continue;for(const contents of visibleContents(runtime.rendition))wireLearningDocument(id,index,contents,runtime.book?.spine.get(contents.sectionIndex)?.href??'');}});
  const scrollPanel=(panel:HTMLElement,list:()=>HTMLElement|null)=>panel.addEventListener('wheel',event=>{
    if(!panel.classList.contains('visible'))return;
    event.preventDefault();event.stopPropagation();const target=list();if(target)target.scrollTop+=event.deltaY*(event.deltaMode===1?18:event.deltaMode===2?target.clientHeight:1);
  },{passive:false});
  scrollPanel(drawer,()=>libraryList);
  scrollPanel(navigationPanel,()=>navigationPanel.querySelector<HTMLElement>(navigationTab==='contents'?'.chapter-list':'.book-search-results'));

  wireReadingNavigation();
  window.addEventListener('reader-study-open',()=>{
    closeJumpDialog();closeFigureViewer();closeBookNavigation();setReadingSettings(false);closeDrawer();setAnnotationPanelVisible(false);
  });

  libraryHandle.addEventListener("click", () => {
    if (drawer.classList.contains("visible")) closeDrawer();
    else openDrawer();
  });

  app.querySelector(".library-toggle")?.addEventListener("click", () => {
    if (drawer.classList.contains("visible")) closeDrawer();
    else openDrawer();
  });
  app.querySelector(".drawer-close")?.addEventListener("click", closeDrawer);
  app.querySelector(".nav-back")?.addEventListener("click", () => navigateActive("prev"));
  app.querySelector(".nav-forward")?.addEventListener("click", () => navigateActive("next"));
  app.querySelector(".nav-group-back")?.addEventListener("click", () =>
    navigateActive("prev", "group"),
  );
  app.querySelector(".nav-group-forward")?.addEventListener("click", () =>
    navigateActive("next", "group"),
  );
  app.querySelector(".jump-toggle")?.addEventListener("click", () => openJumpDialog());
  app.querySelector(".jump-close")?.addEventListener("click", closeJumpDialog);
  app.querySelector(".jump-cancel")?.addEventListener("click", closeJumpDialog);
  jumpDialog.addEventListener("cancel", (event) => {
    event.preventDefault();
    closeJumpDialog();
  });
  jumpDialog.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      closeJumpDialog();
      return;
    }
    if (event.key !== "Tab") return;
    const controls = Array.from(jumpDialog.querySelectorAll<HTMLElement>("button:not(:disabled), input:not(:disabled)"));
    if (!controls.length) return;
    const index = controls.indexOf(document.activeElement as HTMLElement);
    if (index < 0 || (event.shiftKey && index === 0) || (!event.shiftKey && index === controls.length - 1)) {
      event.preventDefault();
      controls[event.shiftKey ? controls.length - 1 : 0].focus();
    }
  });
  jumpDialog.addEventListener("click", (event) => {
    if (event.target === jumpDialog) closeJumpDialog();
  });
  jumpForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const index = snapshot.session.activePane;
    const requested = Number(jumpInput.value);
    if (!Number.isFinite(requested) || requested < 1) {
      jumpInput.focus();
      return;
    }
    closeJumpDialog();
    void jumpToPage(index, requested);
  });
  app.querySelector(".font-down")?.addEventListener("click", () => changeFont(-10));
  app.querySelector(".font-up")?.addEventListener("click", () => changeFont(10));
  app.querySelector(".theme-cycle")?.addEventListener("click", cycleTheme);
  annotationToggle.addEventListener("click", () =>
    setAnnotationPanelVisible(!annotationPanel.classList.contains("visible")),
  );
  app.querySelector(".annotation-close")?.addEventListener("click", () =>
    setAnnotationPanelVisible(false),
  );
  app.querySelector(".apply-text-mark")?.addEventListener("click", applyPendingTextMark);
  app.querySelector(".annotation-undo")?.addEventListener("click", undoLastAnnotation);
  app.querySelectorAll<HTMLButtonElement>(".format-toggle").forEach((button) => {
    button.addEventListener("click", () => {
      const format = button.dataset.format;
      if (!format || !["highlight", "bold", "underline", "wave", "box"].includes(format)) return;
      const key = format as keyof AnnotationTextStyle;
      if (activeFormats.has(key)) activeFormats.delete(key);
      else activeFormats.add(key);
      const selected = activeFormats.has(key);
      button.classList.toggle("selected", selected);
      button.setAttribute("aria-pressed", String(selected));
    });
  });
  app.querySelectorAll<HTMLButtonElement>(".annotation-tool").forEach((button) => {
    button.addEventListener("click", () => {
      const tool = button.dataset.tool;
      if (tool === "read" || tool === "text" || tool === "pen" || tool === "eraser") {
        setAnnotationTool(tool);
      }
    });
  });
  annotationList.addEventListener("click", (event) => {
    const target = event.target as HTMLElement;
    const item = target.closest<HTMLElement>(".annotation-list-item");
    const annotationId = item?.dataset.annotationId;
    const bookId = runtimes[snapshot.session.activePane].bookId;
    if (!annotationId || !bookId) return;
    if (target.closest(".annotation-delete")) deleteBookAnnotation(bookId, annotationId);
    else if (target.closest(".annotation-jump")) jumpToAnnotation(annotationId);
  });
  app.querySelector(".fullscreen-toggle")?.addEventListener("click", () => {
    void toggleFullscreen().catch((error) => showToast(`无法切换全屏：${String(error)}`, "error"));
  });
  app.querySelector(".add-books")?.addEventListener("click", () => void importSelectedBooks());
  app.querySelector(".add-folder")?.addEventListener("click", () => void importFolder());
  app.querySelector(".refresh-library")?.addEventListener("click", () => void refreshLibrary());

  searchInput.addEventListener("input", () => renderLibrary(searchInput.value));
  libraryList.addEventListener("click", (event) => {
    const target = event.target as HTMLElement;
    const item = target.closest<HTMLElement>(".library-book");
    const bookId = item?.dataset.bookId;
    if (!bookId) return;
    if (target.closest(".book-new-pane")) {
      openBookInNewPane(bookId);
      closeDrawer();
    } else if (target.closest(".book-main")) {
      openBookInNewPane(bookId);
      closeDrawer();
    }
  });

  readerGrid.addEventListener("pointerdown", (event) => {
    const pane = (event.target as HTMLElement).closest<HTMLElement>(".reader-pane");
    if (pane) setActivePane(Number(pane.dataset.paneIndex));
    if((event.target as Element).closest('.pane-drag-handle,.workspace-divider'))setToolsVisible(false);
  });
  readerGrid.addEventListener("pointerdown", handleAnnotationPointerDown);
  readerGrid.addEventListener("input", (event) => {
    const target = event.target as HTMLElement;
    if (target.matches(".free-note-content")) updateFreeTextFromEditor(target);
  });
  readerGrid.addEventListener("click", (event) => {
    const target = event.target as HTMLElement;
    const pane = target.closest<HTMLElement>(".reader-pane");
    const index = Number(pane?.dataset.paneIndex);
    if (!Number.isFinite(index)) return;
    if (target.closest(".page-hotspot-group-prev")) {
      void navigate(index, "prev", "group");
    } else if (target.closest(".page-hotspot-prev")) {
      void navigate(index, "prev");
    } else if (target.closest(".page-hotspot-group-next")) {
      void navigate(index, "next", "group");
    } else if (target.closest(".page-hotspot-next")) {
      void navigate(index, "next");
    } else if (target.closest(".pane-close")) {
      closeBook(index);
    } else if (target.closest('.pane-focus')) {
      workspace.focus(index);
    } else if (target.closest(".page-jump-button")) {
      openJumpDialog(index);
    } else if (target.closest(".chapter-boundary-prev,.chapter-boundary-next")) {
      openBoundaryChapter(index, target.closest<HTMLButtonElement>(".chapter-boundary-prev,.chapter-boundary-next")!);
    } else if (target.closest('.retry-book')&&runtimes[index].bookId) {
      void openBook(runtimes[index].bookId!,index);
    } else if (target.closest(".empty-library-button")) {
      if (!isDesktop && !snapshot.books.length) void showCatalogue();
      else openDrawer();
    }
  });
  readerGrid.addEventListener("change", (event) => {
    const target = event.target as HTMLElement;
    if (!target.matches(".pane-page-mode select,.pane-reading-mode select")) return;
    const pane = target.closest<HTMLElement>(".reader-pane");
    const index = Number(pane?.dataset.paneIndex);
    if (!Number.isFinite(index)) return;
    setActivePane(index);
    if(target.matches('.pane-reading-mode select')){void setReadingMode(index,(target as HTMLSelectElement).value==='scroll'?'scroll':'paged');(target as HTMLSelectElement).blur();return;}
    setBookPageMode(index, Number((target as HTMLSelectElement).value));
    // Do not leave keyboard focus inside the select: native left/right would
    // otherwise change 4 pages into 3/5 pages instead of turning the book.
    (target as HTMLSelectElement).blur();
  });

  window.addEventListener("keydown", (event) => {
    if (learningIsOpen()) return;
    if (jumpDialog.open) {
      if (event.key === "Escape") { event.preventDefault(); closeJumpDialog(); }
      return;
    }
    if(figureDialog?.open){if(event.key==='Escape'){event.preventDefault();closeFigureViewer();}return;}
    if(document.querySelector('dialog[open]'))return;
    if (event.key === "Escape") {
      event.preventDefault();
      if (figureDialog?.open) closeFigureViewer();
      else if (navigationPanel.classList.contains("visible")) closeBookNavigation();
      else if (settingsPanel.classList.contains("visible")) setReadingSettings(false);
      else if (annotationPanel.classList.contains("visible")) setAnnotationPanelVisible(false);
      else if(drawer.classList.contains('visible'))closeDrawer();
      else if(app.querySelector('.workspace-panel')?.classList.contains('visible'))workspace.showPanel(false);
      else if(toolsVisible)setToolsVisible(false);
      else workspace.unfocus();
      return;
    }
    if(event.key==='F8'){event.preventDefault();setToolsVisible(!toolsVisible);return;}
    if(event.ctrlKey&&event.shiftKey&&event.key.toLowerCase()==='f'){event.preventDefault();workspace.focus(snapshot.session.activePane);return;}
    if (event.altKey && event.key === "ArrowLeft") { event.preventDefault(); returnToPreviousPosition(); return; }
    if (event.ctrlKey && event.key.toLowerCase() === "f") { event.preventDefault(); openBookNavigation("search"); return; }
    if (event.ctrlKey && event.key.toLowerCase() === "t") { event.preventDefault(); openBookNavigation("contents"); return; }
    if (event.key === "F11") {
      event.preventDefault();
      void toggleFullscreen().catch((error) => showToast(`无法切换全屏：${String(error)}`, "error"));
      return;
    }
    const target = event.target as HTMLElement | null;
    if (event.ctrlKey && event.key.toLowerCase() === "n") {
      event.preventDefault();
      setAnnotationPanelVisible(!annotationPanel.classList.contains("visible"));
      return;
    }
    if (
      target instanceof Element && target.matches(".pane-page-mode select") &&
      (event.key === "ArrowLeft" || event.key === "ArrowRight")
    ) {
      event.preventDefault();
      (target as HTMLSelectElement).blur();
      navigateActive(event.key === "ArrowLeft" ? "prev" : "next");
      return;
    }
    if (event.ctrlKey && event.key.toLowerCase() === "l") {
      event.preventDefault();
      openDrawer();
      searchInput.focus();
      return;
    }
    if (target instanceof Element && target.closest("button, a[href], summary, input, textarea, select, [contenteditable], [role='button'], [role='link'], [role='tab'], [role='menuitem']")) return;
    if (event.key.toLowerCase() === "g") {
      event.preventDefault();
      openJumpDialog();
      return;
    }
    if (event.ctrlKey && /^[1-9]$/.test(event.key)) {
      event.preventDefault();
      const pane=workspace.ids[Number(event.key)-1];if(pane!==undefined)setActivePane(pane);
      return;
    }
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      navigateActive("prev");
      return;
    }
    if (event.key === "ArrowRight") {
      event.preventDefault();
      navigateActive("next");
      return;
    }
    if (event.key === "PageUp" || (event.key === " " && event.shiftKey)) {
      event.preventDefault();
      navigateActive("prev", "group");
      return;
    }
    if (event.key === "PageDown" || (event.key === " " && !event.shiftKey)) {
      event.preventDefault();
      navigateActive("next", "group");
      return;
    }
  });

  window.addEventListener("pointermove", (event) => {
    if (drawingDraft) extendDrawing(event);
    if (noteDragState) moveNote(event);
  });
  window.addEventListener("pointerup", (event) => {
    if (drawingDraft) finishDrawing(event);
    if (noteDragState) finishNoteDrag(event);
  });
  window.addEventListener("pointercancel", (event) => {
    if (drawingDraft) finishDrawing(event);
    if (noteDragState) finishNoteDrag(event);
  });
}

function readingFontFamily(index=snapshot.session.activePane): string {
  return readingPreferences(index).readerFont === "sans"
    ? '"Segoe UI", "Noto Sans SC", "Microsoft YaHei", sans-serif'
    : 'Georgia, "Noto Serif SC", "Source Han Serif SC", SimSun, serif';
}
function fitFlowTablePreview(element:HTMLTableElement,fitKey:string,usableWidth:number,usableHeight:number,columnWidth:number):boolean {
  if(element.dataset.comfortableFitKey===fitKey)return element.classList.contains('reader-flow-table');
  element.style.removeProperty('zoom');
  const rows=Array.from(element.rows);
  // Total row height distinguishes a long table from a wide, short relation.
  // Fragment rectangles span several columns, so their union is not its width.
  const long=rows.reduce((height,row)=>height+row.offsetHeight,0)>usableHeight;
  element.classList.toggle('reader-flow-table',long);
  if(!long)return false;
  element.dataset.comfortableFitKey=fitKey;
  element.style.removeProperty('max-height');element.style.removeProperty('transform-origin');
  const width=Math.max(element.scrollWidth,...Array.from(element.getClientRects()).map(rect=>rect.width));
  if(width>columnWidth+2){
    element.style.zoom=String(Math.max(.05,Math.min(1,usableWidth/width)));
  }
  // A readable continued table still has a useful full-table reading view.
  element.dataset.readerDetail='true';element.dataset.comfortableTableDetail='true';
  element.classList.add('reader-expandable');element.tabIndex=0;element.title='点按查看完整表格';
  element.setAttribute('aria-description','按回车或点按，在完整表格中选择和复制原文');
  return true;
}
function fitAlgorithmPreview(element:HTMLElement,fitKey:string,usableWidth:number,columnWidth:number):void {
  if(element.dataset.comfortableFitKey===fitKey)return;
  element.style.removeProperty("zoom");element.dataset.comfortableFitKey=fitKey;
  const widths=Array.from(element.getClientRects()).map(rect=>rect.width);
  const width=Math.max(0,...widths);
  // Narrow overview columns cannot contain deeply indented code at the
  // normal intrinsic table width. Fit width only; keep semantic row flow.
  if(width>columnWidth+2) {
    element.style.zoom=String(Math.max(.05,Math.min(1,usableWidth/width)));
    element.dataset.readerDetail="true";element.classList.add("reader-expandable");element.tabIndex=0;
    element.title="点按放大阅读算法";
  }
}
function fitFlowCodePreview(element: HTMLElement, fitKey: string, usableWidth: number): void {
  if (element.dataset.comfortableFitKey === fitKey) return;
  element.dataset.comfortableFitKey = fitKey;
  element.style.removeProperty("zoom");
  element.style.removeProperty("max-height");
  element.style.removeProperty("transform-origin");
  element.classList.add("reader-flow-code");
  const code = element.querySelector("code") ?? element;
  const view = element.ownerDocument.defaultView;
  if (!view) return;
  const codeStyle = view.getComputedStyle(code);
  const boxStyle = view.getComputedStyle(element);
  const canvas = document.createElement("canvas");
  const measure = canvas.getContext("2d");
  if (!measure) return;
  measure.font = codeStyle.font || `${codeStyle.fontWeight} ${codeStyle.fontSize} ${codeStyle.fontFamily}`;
  const tabSize = Math.max(1, Math.min(8, Number.parseInt(codeStyle.tabSize, 10) || 4));
  let widestLine = 0;
  for (const line of (code.textContent ?? "").split(/\r\n|\r|\n/)) {
    widestLine = Math.max(widestLine, measure.measureText(line.replace(/\t/g, " ".repeat(tabSize))).width);
  }
  const padding = (Number.parseFloat(boxStyle.paddingLeft) || 0) + (Number.parseFloat(boxStyle.paddingRight) || 0);
  const available = Math.max(16, usableWidth - padding);
  if (widestLine > available + 2) {
    element.dataset.readerDetail = "true";
    element.dataset.comfortableCodeDetail = "true";
    element.classList.add("reader-expandable");
    element.tabIndex = 0;
    element.title = "点按在宽屏详情中阅读、选择并复制完整代码";
  } else if (element.dataset.comfortableCodeDetail === "true") {
    delete element.dataset.readerDetail;
    delete element.dataset.comfortableCodeDetail;
    element.classList.remove("reader-expandable");
    element.removeAttribute("tabindex");
    if (element.title === "点按在宽屏详情中阅读、选择并复制完整代码") element.removeAttribute("title");
  }
}
function readingThemeCss(index=snapshot.session.activePane):string {
  return Object.entries(themeRules(snapshot.session.theme,index)).map(([selector,properties])=>`${selector}{${Object.entries(properties).map(([key,value])=>`${key}:${value}`).join(";")}}`).join("\n");
}
function scrollReadingCss(index:number):string{
  if(runtimes[index]?.readingMode!=='scroll')return '';
  const width=Math.max(28,Math.min(64,readingPreferences(index).contentWidth));
  const contentWidth=Math.max(8,width-2.8).toFixed(1);
  const imageHeight=Math.max(180,Math.round((runtimes[index].viewportHeight||window.innerHeight)*.72));
  return `\nhtml{overflow:visible!important}body{max-width:${width}em!important;margin:0 auto!important;padding:1.8em 1.4em 3em!important;column-rule:none!important;overflow-wrap:break-word}pre{max-width:100%!important;overflow-x:auto;white-space:pre-wrap}table{display:block!important;max-width:100%!important;overflow-x:auto!important;table-layout:auto!important;scrollbar-width:thin}th.reader-short-table-header{white-space:nowrap!important}figure{max-width:100%!important}img,svg,video{max-width:min(calc(100vw - 2.8em),${contentWidth}em)!important;min-width:0!important;height:auto!important}figure img,figure.reader-flow-figure img{max-width:min(calc(100vw - 2.8em),${contentWidth}em)!important;max-height:${imageHeight}px!important;width:auto!important;object-fit:contain!important}.math-block,.math-display,.MathJax_Display,.katex-display{max-width:100%!important;overflow-x:auto!important;overflow-y:auto!important;overscroll-behavior-x:contain;scrollbar-width:thin;padding-bottom:.25em!important}h1{font-size:1.7em!important}h2{font-size:1.3em!important}`;
}
function prepareSectionForLayout(index:number, output:string, section:{output:string}):void {
  // epub.js fires rendition content hooks after the first size measurement.
  // Late font changes in a prepended section otherwise move the scroll origin
  // by its temporary, unscaled width and can restore an earlier chapter.
  // Earlier serialize hooks may already have replaced archive image/CSS URLs.
  // Their current output is authoritative; the hook's first argument is stale.
  const doc=new DOMParser().parseFromString(section.output || output,"application/xhtml+xml");
  if(doc.querySelector("parsererror"))return;
  const original=runtimes[index].book;if(original)restoreSourceStrong(doc,sourceStrong.get(original)??new Set());
  protectHeadingWords(doc);
  const head=doc.querySelector("head");if(!head)return;
  let sheet=doc.getElementById("comfortable-reader-theme");
  if(!sheet){sheet=doc.createElementNS("http://www.w3.org/1999/xhtml","style");sheet.id="comfortable-reader-theme";head.append(sheet);}
  sheet.textContent=readingThemeCss(index)+scrollReadingCss(index)+`\nbody{--reader-logical-height:${runtimes[index].layoutHeight}px;font-size:${18*runtimes[index].effectiveFontScale/100}px}h1{font-size:clamp(1.25em,calc(var(--reader-logical-height)*.085),1.7em)!important;text-wrap:balance!important}`;
  section.output=new XMLSerializer().serializeToString(doc);
}
function applyReadingTheme(contents: EpubContents): void {
  // epub.js addStylesheetRules appends even when the key already exists.
  // Replacing one owned sheet prevents old palettes and fonts winning later
  // in the cascade, including when returning to the publisher's font.
  const frame=contents.window.frameElement as Element|null;
  const index=Number(frame?.closest<HTMLElement>('.reader-pane')?.dataset.paneIndex??-1);
  protectHeadingWords(contents.document);
  const css=readingThemeCss(index>=0?index:snapshot.session.activePane)+(index>=0?scrollReadingCss(index):'');
  let sheet=contents.document.getElementById("comfortable-reader-theme");
  if(!sheet) {sheet=contents.document.createElement("style");sheet.id="comfortable-reader-theme";contents.document.head.append(sheet);}
  sheet.textContent=css;
  if(index>=0)sheet.textContent+=`\nbody{--reader-logical-height:${runtimes[index].layoutHeight}px}h1{font-size:clamp(1.25em,calc(var(--reader-logical-height)*.085),1.7em)!important;text-wrap:balance!important}`;
  contents.document.documentElement.dataset.readerTheme=snapshot.session.theme;
}

let navigationTab: "contents" | "search" = "contents";
let searchGeneration = 0;
let navigationBookId: string | null = null;
let searchHits: Array<{ cfi: string; excerpt: string; chapter: string }> = [];
const searchHighlights = new WeakMap<Rendition,string>();
const paneOperations=Array.from({length:MAX_PANES},()=>({generation:-1,pending:Promise.resolve()}));
const layoutRequests=[0,0,0,0];
const indexingBooks=new WeakSet<EpubBook>();
const navigationSequence=[0,0,0,0];
const navigationHistory=Array.from({length:MAX_PANES},()=>({bookId:null as string|null,entries:[] as string[]}));
const navigationPanel = requireElement<HTMLElement>(app, ".book-navigation");
const settingsPanel = requireElement<HTMLElement>(app, ".reading-settings");
const bookSearchInput = requireElement<HTMLInputElement>(app, ".book-search-input");
const chapterList = requireElement<HTMLElement>(app, ".chapter-list");
const searchResults = requireElement<HTMLElement>(app, ".book-search-results");
const navigationStatus = requireElement<HTMLElement>(app, ".navigation-status");
let navigationReturnFocus: HTMLElement | null = null;
let settingsReturnFocus: HTMLElement | null = null;

interface SearchableSection {
  href: string;
  linear?: string | boolean;
  document?: Document;
  load: (load: unknown) => Promise<unknown>;
  cfiFromRange: (range: Range) => string;
}
function findNormalizedText(section: SearchableSection, query: string): Array<{ cfi: string; excerpt: string }> {
  const doc=section.document;
  if(!doc)return [];
  const root=doc.querySelector("body")??doc.documentElement;
  const walker=doc.createTreeWalker(root,NodeFilter.SHOW_TEXT);
  const chars:string[]=[];
  const positions:Array<{node:Text;start:number;end:number}>=[];
  let lastBlock:Element|null=null,node:Node|null;
  while((node=walker.nextNode())) {
    const parent=node.parentElement;
    if(!parent||parent.closest("script,style,annotation,annotation-xml"))continue;
    const block=parent.closest("p,li,h1,h2,h3,h4,h5,h6,td,th,figcaption");
    if(lastBlock&&block!==lastBlock&&chars.length&&chars[chars.length-1]!==" ") {chars.push(" ");positions.push(positions[positions.length-1]);}
    lastBlock=block;
    const text=node.textContent??"";
    for(let offset=0;offset<text.length;) {
      const value=String.fromCodePoint(text.codePointAt(offset)!);
      const normalized=value.normalize("NFKC").toLocaleLowerCase();
      for(const char of normalized) {
        const safe=/\s/u.test(char)?" ":char;
        if(safe===" "&&chars[chars.length-1]===" ")continue;
        // Search normalization expands ligatures while every result retains
        // an offset into the unchanged EPUB text and its original CFI.
        for(let codeUnit=0;codeUnit<safe.length;codeUnit++) {
          chars.push(safe[codeUnit]);positions.push({node:node as Text,start:offset,end:offset+value.length});
        }
      }
      offset+=value.length;
    }
  }
  const text=chars.join("");
  const needle=query.normalize("NFKC").toLocaleLowerCase().replace(/\s+/gu," ");
  const found:Array<{cfi:string;excerpt:string}>=[];
  let at=-1;
  while((at=text.indexOf(needle,at+1))>=0&&found.length<150) {
    const first=positions[at],last=positions[at+needle.length-1];
    if(!first||!last)continue;
    const range=doc.createRange();range.setStart(first.node,first.start);range.setEnd(last.node,last.end);
    found.push({cfi:section.cfiFromRange(range),excerpt:(at>55?"…":"")+text.slice(Math.max(0,at-55),at+needle.length+85)+(at+needle.length+85<text.length?"…":"")});
  }
  return found;
}

function closeBookNavigation(): void {
  const wasVisible = navigationPanel.classList.contains("visible");
  const hadFocus = navigationPanel.contains(document.activeElement);
  navigationPanel.classList.remove("visible");
  navigationPanel.inert = true;
  navigationPanel.setAttribute("aria-hidden", "true");
  app.querySelector(".contents-toggle")?.setAttribute("aria-expanded", "false");
  app.querySelector(".book-search-toggle")?.setAttribute("aria-expanded", "false");
  if (wasVisible && hadFocus) {
    const fallback = app.querySelector<HTMLElement>(toolsVisible?(navigationTab === "search" ? ".book-search-toggle" : ".contents-toggle"):'.tools-toggle');
    const trigger = navigationReturnFocus?.isConnected && !navigationReturnFocus.closest("[inert]") ? navigationReturnFocus : fallback;
    trigger?.focus({ preventScroll: true });
  }
  navigationReturnFocus = null;
}
function setReadingSettings(visible: boolean): void {
  const wasVisible = settingsPanel.classList.contains("visible");
  const hadFocus = settingsPanel.contains(document.activeElement);
  if (visible && !wasVisible) {
    const starter = document.activeElement;
    settingsReturnFocus = starter instanceof HTMLElement && starter !== document.body ? starter : app.querySelector<HTMLElement>(".reading-settings-toggle");
  }
  if (visible) { closeDrawer(); closeBookNavigation(); setAnnotationPanelVisible(false);workspace?.showPanel(false,false); }
  settingsPanel.classList.toggle("visible", visible);
  settingsPanel.inert = !visible;
  settingsPanel.setAttribute("aria-hidden", String(!visible));
  app.querySelector(".reading-settings-toggle")?.setAttribute("aria-expanded", String(visible));
  syncReadingSettings();
  if (visible && !wasVisible) requireElement<HTMLSelectElement>(settingsPanel, ".reading-mode").focus();
  if (!visible && wasVisible) {
    if (hadFocus) {
      const trigger = settingsReturnFocus?.isConnected && !settingsReturnFocus.closest("[inert]")
        ? settingsReturnFocus : app.querySelector<HTMLElement>(".reading-settings-toggle");
      trigger?.focus({ preventScroll: true });
    }
    settingsReturnFocus = null;
  }
}
function openBookNavigation(tab: "contents" | "search"): void {
  if (!navigationPanel.classList.contains("visible") || !navigationPanel.contains(document.activeElement)) {
    const starter = document.activeElement;
    navigationReturnFocus = starter instanceof HTMLElement && starter !== document.body
      ? starter : app.querySelector<HTMLElement>(tab === "search" ? ".book-search-toggle" : ".contents-toggle");
  }
  closeDrawer(); setReadingSettings(false);setAnnotationPanelVisible(false);workspace?.showPanel(false,false);
  navigationTab = tab;
  navigationPanel.inert = false;
  navigationPanel.classList.add("visible");
  navigationPanel.setAttribute("aria-hidden", "false");
  app.querySelector(".contents-toggle")?.setAttribute("aria-expanded", String(tab === "contents"));
  app.querySelector(".book-search-toggle")?.setAttribute("aria-expanded", String(tab === "search"));
  requireElement<HTMLElement>(navigationPanel, ".navigation-heading").textContent = tab === "contents" ? "本书目录" : "搜索本书";
  requireElement<HTMLElement>(navigationPanel, ".book-search-form").hidden = tab !== "search";
  chapterList.hidden = tab !== "contents";
  searchResults.hidden = tab !== "search";
  navigationPanel.querySelectorAll<HTMLElement>("[data-navigation-tab]").forEach(button => {
    const selected = button.dataset.navigationTab === tab;
    button.classList.toggle("selected", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
  renderChapterList();
  if (tab === "search") { bookSearchInput.focus(); bookSearchInput.select(); }
  else requireElement<HTMLButtonElement>(navigationPanel, '[data-navigation-tab="contents"]').focus();
}
function renderChapterList(): void {
  const runtime = runtimes[snapshot.session.activePane];
  if (!runtime) return;
  if (navigationBookId !== runtime.bookId) {
    navigationBookId = runtime.bookId;
    searchGeneration++;
    searchHits = [];
    searchResults.replaceChildren();
    bookSearchInput.value = "";
  }
  type TocItem = { label: string; href: string; subitems?: TocItem[] };
  const toc = runtime.book?.navigation?.toc as TocItem[] | undefined;
  const render = (items: TocItem[], depth = 0): string => items.map(item => {
    const label=item.label.trim();
    const inferredDepth=depth||(/^[★\s]*[\dA-D]+\.\d/.test(label)?1:0);
    return `<div class="chapter-entry"><button type="button" data-chapter-href="${escapeHtml(item.href)}" style="--depth:${inferredDepth}">${escapeHtml(label)}</button>${item.subitems?.length ? render(item.subitems, depth + 1) : ""}</div>`;
  }).join("");
  chapterList.innerHTML = toc?.length ? render(toc) : '<p class="navigation-empty">这本书没有独立目录。</p>';
  markCurrentToc();
  if (navigationTab === "contents") navigationStatus.textContent = getBook(runtime.bookId)?.title ?? "请先打开一本书";
  else if (!searchHits.length) navigationStatus.textContent = "搜索整本书，结果直接定位到原文。";
}
function markCurrentToc():void{
  const buttons=Array.from(chapterList.querySelectorAll<HTMLButtonElement>('[data-chapter-href]'));
  for(const button of buttons)button.removeAttribute('aria-current');
  const runtime=runtimes[snapshot.session.activePane];if(!runtime.book||!runtime.rendition||!runtime.cfi)return;
  try{
    const section=runtime.book.spine.get(runtime.cfi);const contents=visibleContents(runtime.rendition).find(c=>c.sectionIndex===section?.index);if(!contents)return;
    let selected:HTMLButtonElement|null=null,first:HTMLButtonElement|null=null,best:string|null=null;
    for(const button of buttons){const href=button.dataset.chapterHref!;if(runtime.book.spine.get(href)?.index!==section?.index)continue;
      first??=button;
      const id=href.split('#')[1];const node=id?contents.document.getElementById(decodeURIComponent(id)):contents.document.body;if(!node)continue;
      const range=contents.document.createRange();range.selectNodeContents(node);range.collapse(true);const cfi=contents.cfiFromRange(range,'reader-annotation');const comparator=contents.epubcfi;
      if(comparator.compare(cfi,runtime.cfi)<=0&&(!best||comparator.compare(cfi,best)>=0)){selected=button;best=cfi;}
    }
    (selected??first)?.setAttribute('aria-current','location');
  }catch{/* Navigation still works when a publisher target has no comparable heading. */}
}
async function searchCurrentBook(): Promise<void> {
  const query = bookSearchInput.value.trim();
  const runtime = runtimes[snapshot.session.activePane];
  const book = runtime.book;
  const generation = ++searchGeneration;
  searchHits = []; searchResults.replaceChildren();
  if (!book || query.length < 2) { navigationStatus.textContent = "请输入至少两个字符。"; return; }
  const sections = (book.spine as unknown as { spineItems: SearchableSection[] }).spineItems.filter(section => section.linear !== false && section.linear !== "no");
  const headings = new Map<string,string>();
  const walk = (items: Array<{href: string; label: string; subitems?: Array<any>}>) => { for (const item of items) { const key = item.href.split("#")[0]; if (!headings.has(key)) headings.set(key,item.label); if(item.subitems)walk(item.subitems); } };
  walk(book.navigation.toc);
  let unreadable = 0;
  for (let i = 0; i < sections.length; i++) {
    if (generation !== searchGeneration || runtime.book !== book) return;
    const section = sections[i];
    try {
      await section.load(book.load.bind(book));
      if (generation !== searchGeneration || runtime.book !== book) return;
      const hits = findNormalizedText(section,query);
      const seen = new Set(searchHits.map(hit => hit.cfi));
      for (const hit of hits) {
        if (seen.has(hit.cfi)) continue;
        seen.add(hit.cfi);
        searchHits.push({...hit, chapter: headings.get(section.href) ?? `第 ${i+1} 节`});
        if (searchHits.length >= 150) break;
      }
    } catch { unreadable++; }
    if (i % 4 === 0 || i === sections.length - 1 || searchHits.length >= 150) {
      searchResults.innerHTML = searchHits.map((hit, position)=>`<button type="button" class="search-result" data-search-hit="${position}"><strong>${escapeHtml(hit.chapter)}</strong><span>${escapeHtml(hit.excerpt)}</span></button>`).join("");
      navigationStatus.textContent = `已检索 ${i+1}/${sections.length} 节 · ${searchHits.length} 处匹配`;
      await new Promise<void>(resolve=>window.setTimeout(resolve,0));
    }
    if (searchHits.length >= 150) break;
  }
  if (generation !== searchGeneration) return;
  navigationStatus.textContent = (searchHits.length >= 150 ? "显示前 150 处匹配，请用更具体的词缩小范围。" : `找到 ${searchHits.length} 处匹配`) + (unreadable ? `；${unreadable} 节检索失败。` : "");
}
function enqueuePaneOperation(index:number,operation:()=>Promise<void>):Promise<void> {
  const entry=paneOperations[index],generation=runtimes[index].generation;
  if(entry.generation!==generation){entry.generation=generation;entry.pending=Promise.resolve();}
  const opening=runtimes[index].opening;
  const task=entry.pending.catch(()=>{}).then(async()=>{if(opening)await opening;if(runtimes[index].generation===generation)await operation();});
  entry.pending=task;return task;
}
function jumpToBookTarget(target: string, highlight = false, index = snapshot.session.activePane, remember = true): Promise<void> {
  return enqueuePaneOperation(index,async()=>{await performBookJump(index,target,highlight,remember);});
}
async function restoreImportedPosition(target:string,index:number):Promise<boolean>{let result=false;await enqueuePaneOperation(index,async()=>{result=await performBookJump(index,target,false,true);});return result;}
function jumpToScreenPage(index:number,requestedPage:number):Promise<void> {
  return enqueuePaneOperation(index,async()=>{
    const runtime=runtimes[index],rendition=runtime.rendition;if(!rendition)return;
    const target=Math.min(Math.max(1,Math.round(requestedPage)),Math.max(1,runtime.totalPages));
    const internal=rendition as InternalRendition;
    const manager=internal.manager as typeof internal.manager & {settings:{direction?:string};scrollTo:(x:number,y:number,silent:boolean)=>void};
    if(!manager.container)return;
    const history=navigationHistory[index];
    if(history.bookId!==runtime.bookId){history.bookId=runtime.bookId;history.entries=[];}
    if(runtime.cfi&&history.entries[history.entries.length-1]!==runtime.cfi){history.entries.push(runtime.cfi);if(history.entries.length>50)history.entries.shift();}
    updateNavigationHistoryUi();
    runtime.preserveSemanticFocus=false;
    const direction=manager.settings.direction==='rtl'?-1:1;
    const left=manager.container.scrollLeft+(target-runtime.currentPage)*internal._layout.pageWidth*direction;
    manager.scrollTo(left,0,true);
    await new Promise<void>(resolve=>requestAnimationFrame(()=>resolve()));
    await settleBookLocation(index,rendition);
    runtime.layoutAnchorCfi=runtime.cfi;
  });
}
function reflowPane(index:number,force=false):Promise<void> {
  const request=++layoutRequests[index];
  return enqueuePaneOperation(index,async()=>{if(request===layoutRequests[index])await reflowPaneNow(index,force);});
}
function navigate(index:number,direction:"prev"|"next",distance:NavigationDistance="single"):Promise<void> {
  return enqueuePaneOperation(index,async()=>{
    const rendition=runtimes[index].rendition;
    await navigateNow(index,direction,distance);
    if(rendition&&runtimes[index].rendition===rendition)await settleBookLocation(index,rendition);
    runtimes[index].layoutAnchorCfi=runtimes[index].cfi;
  });
}
async function settleRenditionLayout(index:number):Promise<void> {
  const runtime=runtimes[index],rendition=runtime.rendition;
  if(!rendition)return;
  const start=performance.now();let signature="",quietSince=start;
  while(performance.now()-start<650&&runtime.rendition===rendition) {
    await new Promise<void>(resolve=>requestAnimationFrame(()=>resolve()));
    const frames=Array.from(paneElement(index).querySelectorAll<HTMLIFrameElement>(".epub-host iframe"));
    const next=frames.map(frame=>`${frame.clientWidth}:${frame.clientHeight}:${frame.contentDocument?.body?.scrollWidth??0}`).join("|");
    const ready=frames.length>0&&frames.every(frame=>Boolean(frame.contentDocument?.body)&&frame.contentDocument?.fonts.status!=="loading"&&Array.from(frame.contentDocument?.images??[]).every(image=>image.complete));
    if(next!==signature||!ready){signature=next;quietSince=performance.now();}
    else if(performance.now()-quietSince>=65)return;
  }
}
async function settleBookLocation(index:number, rendition:Rendition): Promise<void> {
  await new Promise<void>(resolve=>{
    let timer:number;
    const done=()=>{window.clearTimeout(timer);rendition.off("relocated",done);resolve();};
    rendition.on("relocated",done);
    timer=window.setTimeout(done,1000);
    void rendition.reportLocation();
  });
  if(runtimes[index].rendition!==rendition)return;
}
function setReadingMode(index:number,mode:'paged'|'scroll'):Promise<void>{
  return enqueuePaneOperation(index,async()=>{
    const runtime=runtimes[index],rendition=runtime.rendition;
    if(!runtime.bookId||!rendition||runtime.readingMode===mode)return;
    const previousFocus=runtime.layoutAnchorCfi??runtime.cfi;
    const anchor=previousFocus&&contentCfiVisible(index,previousFocus)
      ? previousFocus : firstVisibleContentCfi(index)??previousFocus;
    const oldMode=runtime.readingMode,oldFocus=runtime.layoutAnchorCfi,oldPreserve=runtime.preserveSemanticFocus,generation=runtime.generation;
    runtime.restoringLocation=true;
    for(const annotation of annotationsForBook(runtime.bookId))if(annotation.kind!=='text-mark'&&!annotation.contentPlacement)captureAnnotationPlacement(index,annotation);
    try{
      runtime.readingMode=mode;runtime.layoutAnchorCfi=anchor;runtime.preserveSemanticFocus=Boolean(anchor);
      const stage=paneElement(index).querySelector<HTMLElement>('.pane-stage')!;
      const available=availableStageSize(stage);runtime.layoutWidth=available.width;runtime.layoutHeight=available.height;
      runtime.actualPageCount=mode==='scroll'?1:runtime.pageMode||automaticPageCountForWidth(available.width,index);
      positionRenditionCanvas(index);applyAdaptiveFontScale(index,paneElement(index).querySelector('.epub-host')!);
      patchMultiPageLayout(rendition,runtime);rendition.flow(mode==='scroll'?'scrolled-continuous':'paginated');
      rendition.resize(runtime.layoutWidth,runtime.layoutHeight);
      if(anchor)await rendition.display(anchor);
      await settleRenditionLayout(index);fitAtomicBlocks(index);
      if(anchor){await rendition.display(anchor);await ensureAnchorVisible(index,anchor);}
      if(runtime.generation!==generation)return;
      ensureBookProgress(runtime.bookId,index).readingMode=mode;
    }catch(e){if(runtime.generation!==generation)return;runtime.readingMode=oldMode;runtime.layoutAnchorCfi=oldFocus;runtime.preserveSemanticFocus=oldPreserve;runtime.actualPageCount=oldMode==='scroll'?1:runtime.pageMode||automaticPageCountForWidth(runtime.layoutWidth,index);patchMultiPageLayout(rendition,runtime);rendition.flow(oldMode==='scroll'?'scrolled-continuous':'paginated');if(anchor)await rendition.display(anchor).catch(()=>{});showToast('阅读方式暂未切换，原位置仍已保留：'+String(e),'error');}
    finally{if(runtime.generation===generation)runtime.restoringLocation=false;}
    if(runtime.generation!==generation)return;
    await settleBookLocation(index,rendition);updatePaneHeader(index);updatePaneProgress(index);updateChapterBoundary(index);savePaneProgress(index,0);scheduleDynamicPageMap(index,0);
  });
}
async function ensureAnchorVisible(index:number,cfi:string):Promise<void>{
  const runtime=runtimes[index],rendition=runtime.rendition;if(!rendition||!cfi.startsWith('epubcfi('))return;
  try{
    const range=rendition.getRange(cfi,'reader-annotation');if(!range)return;
    const frame=Array.from(paneElement(index).querySelectorAll<HTMLIFrameElement>('.epub-host iframe')).find(f=>f.contentDocument===range.startContainer.ownerDocument);if(!frame)return;
    const rect=range.getClientRects()[0]??range.getBoundingClientRect();if(!rect||!Number.isFinite(rect.left)||rect.height<=0)return;
    const host=paneElement(index).querySelector<HTMLElement>('.epub-host')!,view=host.getBoundingClientRect(),frameBox=frame.getBoundingClientRect();
    const scale=frameBox.width/frame.offsetWidth;if(!Number.isFinite(scale)||scale<=0)return;
    const x=frameBox.left+rect.left*scale;
    if(runtime.readingMode==='scroll'){
      const y=frameBox.top+rect.top*scale;const container=(rendition as InternalRendition).manager.container;
      if(container&&(y<view.top||y+rect.height*scale>view.bottom))container.scrollTop+=(y-view.top)/scale-16;
      return;
    }
    if(x>=view.left-.5&&x<view.right-1)return;
    // epub.js floors location/pageWidth. At an exact CSS column boundary a
    // subpixel rounding error can land one column early. Use the actual target
    // and the measured grid; never guess a page from text length.
    const internal=rendition as InternalRendition;
    const manager=internal.manager as typeof internal.manager&{scrollTo:(x:number,y:number,silent:boolean)=>void};
    const container=manager.container,pageWidth=internal._layout.pageWidth;if(!container||!(pageWidth>0))return;
    const destination=Math.round((container.scrollLeft+(x-view.left)/scale)/pageWidth)*pageWidth;
    manager.scrollTo(destination,0,true);await new Promise<void>(resolve=>requestAnimationFrame(()=>resolve()));
  }catch{/* Existing unavailable-anchor handling remains responsible for stale CFIs. */}
}
async function stableTargetCfi(book:EpubBook,target:string): Promise<string> {
  if(target.startsWith("epubcfi(")||!target.includes("#"))return target;
  const [path,fragment]=target.split("#",2);
  const section=book.spine.get(path);
  if(!section)return target;
  await section.load(book.load.bind(book));
  const doc=section.document;
  const element=doc?.getElementById(decodeURIComponent(fragment));
  if(!element)return target;
  const walker=doc.createTreeWalker(element,NodeFilter.SHOW_TEXT|NodeFilter.SHOW_ELEMENT);
  let node:Node|null;
  while((node=walker.nextNode())) {
    const range=doc.createRange();
    if(node.nodeType===Node.TEXT_NODE&&node.textContent?.trim()) {
      const offset=node.textContent.search(/\S/u);range.setStart(node,offset);range.setEnd(node,offset+1);
    } else if(node.nodeType===Node.ELEMENT_NODE&&(node as Element).matches("img,svg"))return section.cfiFromElement(node as Element);
    else continue;
    return section.cfiFromRange(range);
  }
  return target;
}
function updateNavigationHistoryUi():void {
  const index=snapshot.session.activePane,entry=navigationHistory[index];
  const button=app.querySelector<HTMLButtonElement>(".reading-back");
  if(button)button.disabled=entry.bookId!==runtimes[index].bookId||!entry.entries.length;
}
function returnToPreviousPosition():void {
  const index=snapshot.session.activePane,entry=navigationHistory[index];
  if(entry.bookId!==runtimes[index].bookId)return;
  const cfi=entry.entries.pop();updateNavigationHistoryUi();
  if(cfi)void jumpToBookTarget(cfi,false,index,false);
}
async function performBookJump(index:number, target:string, highlight:boolean, remember:boolean): Promise<boolean> {
  const runtime = runtimes[index];
  const request=++navigationSequence[index];
  const pane=paneElement(index);
  const rendition=runtime.rendition;
  const oldAnchor=runtime.cfi,oldFocus=runtime.layoutAnchorCfi,oldPreserve=runtime.preserveSemanticFocus;let succeeded=false;
  closeBookNavigation();
  try {
    if(!rendition)return false;
    const history=navigationHistory[index];
    if(history.bookId!==runtime.bookId){history.bookId=runtime.bookId;history.entries=[];}
    if(remember&&runtime.cfi&&history.entries[history.entries.length-1]!==runtime.cfi){history.entries.push(runtime.cfi);if(history.entries.length>50)history.entries.shift();}
    updateNavigationHistoryUi();
    runtime.restoringLocation=true;
    pane.setAttribute("aria-busy","true");
    const previous=searchHighlights.get(rendition);
    if(previous)rendition.annotations.remove(previous,"highlight");
    const destination=runtime.book?await stableTargetCfi(runtime.book,target):target;
    runtime.layoutAnchorCfi=destination.startsWith('epubcfi(')?destination:null;
    runtime.preserveSemanticFocus=Boolean(runtime.layoutAnchorCfi);
    await rendition.display(destination);
    await settleRenditionLayout(index);
    if(request!==navigationSequence[index]||runtime.rendition!==rendition)return false;
    fitAtomicBlocks(index);
    await settleRenditionLayout(index);
    // Image fitting and long table fragmentation can change a section after
    // the first display. Reassert the requested anchor before saving progress.
    await rendition.display(destination);
    if(request!==navigationSequence[index]||runtime.rendition!==rendition)return false;
    await ensureAnchorVisible(index,destination);
    runtime.restoringLocation=false;
    await settleBookLocation(index,rendition);
    runtime.layoutAnchorCfi=destination.startsWith('epubcfi(')?destination:runtime.cfi;
    if(highlight) {
      rendition.annotations.highlight(target,{},()=>{},"reader-search-hit",{fill:"#d8913d","fill-opacity":"0.32","mix-blend-mode":"multiply"});
      searchHighlights.set(rendition,target);
    }
    succeeded=true;
  }
  catch (error) { if(runtime.rendition===rendition){runtime.layoutAnchorCfi=oldFocus;runtime.preserveSemanticFocus=oldPreserve;if(oldAnchor&&rendition){try{await rendition.display(oldAnchor);await settleRenditionLayout(index);await ensureAnchorVisible(index,oldAnchor);runtime.restoringLocation=false;await settleBookLocation(index,rendition);}catch{pane.classList.add('load-error');}}showToast(`目标尚未打开，原位置与记录已保留：${String(error)}`, "error");} }
  finally {if(request===navigationSequence[index]&&runtime.rendition===rendition){runtime.restoringLocation=false;pane.setAttribute("aria-busy","false");}}
  return succeeded;
}
let figureDialog: HTMLDialogElement | null = null;
let disposeFigureImage:(()=>void)|null=null;
let figureImageGeneration=0;
let figureReturnFocus:HTMLElement|null=null;
function closeFigureViewer(): void { figureImageGeneration++;disposeFigureImage?.();disposeFigureImage=null;figureDialog?.close();figureReturnFocus?.focus({preventScroll:true}); }
function ensureDetailDialog():HTMLDialogElement {
  if (!figureDialog) {
    figureDialog = document.createElement("dialog");
    figureDialog.className = "figure-viewer";
    figureDialog.innerHTML = `<header><span>原图 · 可滚动查看</span><button type="button" class="icon-button" aria-label="关闭插图">${icons.close}</button></header><div class="figure-viewer-content"></div>`;
    figureDialog.querySelector("button")?.addEventListener("click",closeFigureViewer);
    figureDialog.addEventListener('cancel',()=>{figureImageGeneration++;disposeFigureImage?.();disposeFigureImage=null;figureReturnFocus?.focus({preventScroll:true});});
    figureDialog.addEventListener("click",event=>{if(event.target===figureDialog)closeFigureViewer();});
    app.append(figureDialog);
  }
  return figureDialog;
}
function openDetailDialog(dialog:HTMLDialogElement):void {
  dialog.showModal();
  dialog.scrollTop=0;dialog.scrollLeft=0;
  const content=dialog.querySelector<HTMLElement>('.figure-viewer-content');
  if(content){content.scrollTop=0;content.scrollLeft=0;}
}
function showFigureViewer(image: HTMLImageElement): void {
  const figureDialog=ensureDetailDialog();
  disposeFigureImage?.();disposeFigureImage=null;const generation=++figureImageGeneration;figureReturnFocus=image;
  const sourceCaption = image.closest("figure")?.querySelector("figcaption");
  const caption = sourceCaption?.textContent?.trim() ? sourceCaption.cloneNode(true) as HTMLElement : null;
  if (caption) {
    // Keep MathML, emphasis and paragraphs. A textContent copy loses powers,
    // fractions and semantic grouping in real algorithm-book captions.
    for(const node of [caption,...Array.from(caption.querySelectorAll<HTMLElement>('*'))]){
      if(node.matches('script,iframe,object,embed,style')){node.remove();continue;}
      const presentation=['color','font-weight','font-style','text-decoration','white-space'].map(key=>[key,node.style?.getPropertyValue(key)??'']);
      for(const attribute of Array.from(node.attributes))if(attribute.name.toLowerCase().startsWith('on'))node.removeAttribute(attribute.name);
      if(node.hasAttribute('href')){try{const url=new URL(node.getAttribute('href')!,location.href);if(!['http:','https:','mailto:'].includes(url.protocol))node.removeAttribute('href');}catch{node.removeAttribute('href');}}
      node.removeAttribute('style');node.removeAttribute('tabindex');node.removeAttribute('data-reader-detail');node.classList.remove('reader-expandable');
      for(const [key,value] of presentation)if(value)node.style?.setProperty(key,value);
    }
    const originalImages=Array.from(sourceCaption!.querySelectorAll<HTMLImageElement>('img'));
    caption.querySelectorAll<HTMLImageElement>('img').forEach((copy,i)=>{const source=originalImages[i];if(source)copy.src=source.currentSrc||source.src;});
    const frame=image.ownerDocument.defaultView?.frameElement as HTMLElement|null;
    const index=Number(frame?.closest<HTMLElement>('.reader-pane')?.dataset.paneIndex??snapshot.session.activePane);
    caption.addEventListener('click',event=>{
      const anchor=(event.target as Element).closest<HTMLAnchorElement>('a[href]');if(!anchor)return;
      const href=anchor.getAttribute('href')??'';
      if(/^(?:https?:|mailto:)/i.test(href)){anchor.target='_blank';anchor.rel='noopener noreferrer';return;}
      event.preventDefault();if(!href||/^[a-z][a-z\d+.-]*:/i.test(href))return;
      closeFigureViewer();void jumpToBookTarget(href,false,index);
    });
    caption.style.cssText="display:block;max-width:750px;margin:16px auto;padding:14px 16px;font-size:16px;line-height:1.7;text-align:left;color:#25231f;background:#fbfaf6;border:1px solid #b9b2a6;border-radius:6px";
  }
  figureDialog.dataset.kind="image";
  figureDialog.querySelector("header span")!.textContent="插图 · 适配与放大";
  figureDialog.querySelector("button")!.setAttribute("aria-label","关闭插图");
  const stage=document.createElement('div');stage.className='figure-image-stage';
  figureDialog.querySelector(".figure-viewer-content")?.replaceChildren(stage,...(caption?[caption]:[]));
  openDetailDialog(figureDialog);
  void mountImage(stage,image.currentSrc||image.src,image.alt||'原始插图',null,()=>{}).then(dispose=>{if(generation!==figureImageGeneration){dispose();return;}disposeFigureImage=dispose;}).catch(error=>{if(generation===figureImageGeneration)showToast(readableError(error),'error');});
}
function showTextDetail(element:Element,index:number):void {
  figureImageGeneration++;disposeFigureImage?.();disposeFigureImage=null;figureReturnFocus=element as HTMLElement;
  const dialog=ensureDetailDialog();
  const label=element.classList.contains("reader-algorithm")?"算法":element.tagName.toLowerCase()==="table"?"表格":element.tagName.toLowerCase()==="pre"?"代码":"公式";
  dialog.dataset.kind="text";
  dialog.querySelector("header span")!.textContent=`${label} · 可选择、复制原文`;
  dialog.querySelector("button")!.setAttribute("aria-label","关闭细节视图");
  const isCode=element.tagName.toLowerCase()==="pre";
  const copy=isCode?document.createElement("pre"):element.cloneNode(true) as HTMLElement;
  if(isCode)copy.textContent=element.textContent;
  for(const node of [copy,...Array.from(copy.querySelectorAll<HTMLElement>("*"))]) {
    if(node.matches("script,iframe,object,embed,style")){node.remove();continue;}
    for(const attribute of Array.from(node.attributes))if(attribute.name.toLowerCase().startsWith("on"))node.removeAttribute(attribute.name);
    node.removeAttribute("tabindex");node.removeAttribute("data-reader-detail");node.classList.remove("reader-expandable");
    if(node.style){node.style.removeProperty("zoom");node.style.removeProperty("max-height");node.style.removeProperty("transform-origin");}
  }
  if (isCode) {
    copy.style.cssText += ";width:max-content;min-width:100%;max-width:none;overflow:visible;white-space:pre;font:16px/1.5 'Cascadia Code',Consolas,monospace;text-align:left;tab-size:4";
  }
  copy.addEventListener("click",event=>{
    const anchor=(event.target as Element).closest("a[href]") as HTMLAnchorElement|null;
    if(!anchor)return;
    const href=anchor.getAttribute("href")??"";
    if(/^(?:https?:|mailto:)/i.test(href)){anchor.target="_blank";anchor.rel="noopener noreferrer";return;}
    event.preventDefault();
    if(/^[a-z][a-z\d+.-]*:/i.test(href))return;
    closeFigureViewer();void jumpToBookTarget(href,false,index);
  });
  const content=dialog.querySelector<HTMLElement>(".figure-viewer-content")!;
  content.style.setProperty("--detail-font",readingFontFamily());
  content.replaceChildren(copy);openDetailDialog(dialog);
}
const wheelState = Array.from({length:MAX_PANES},()=>({sum:0,last:0,turned:0}));
function wireBookDocument(index: number, contents: EpubContents): void {
  const frame=contents.window.frameElement as HTMLIFrameElement|null;if(frame)frame.title=`${getBook(runtimes[index].bookId)?.title??'当前书籍'} · ${contents.document.title||'正文'}`;
  if (runtimes[index].bookId) wireLearningDocument(runtimes[index].bookId!, index, contents, runtimes[index].book?.spine.get(contents.sectionIndex)?.href ?? "");
  const doc = contents.document;
  fitInlineStops(doc);
  for(const image of Array.from(doc.images)) {
    if(image.closest('a'))continue;
    image.tabIndex=0;image.setAttribute('role','button');
    image.setAttribute('aria-label',image.alt?'查看原图：'+image.alt:'查看原图');
    image.title=image.title||'查看原图';image.style.cursor='zoom-in';
  }

  doc.addEventListener("click",event=>{
    const anchor=(event.target as Element).closest("a[href]");
    const raw=anchor?.getAttribute("href");
    if(!raw||/^(?:[a-z][a-z\d+.-]*:|\/\/)/i.test(raw))return;
    const book=runtimes[index].book;if(!book)return;
    const base=doc.querySelector("base")?.getAttribute("href")??book.spine.get(contents.sectionIndex).href;
    const resolved=new URL(raw,new URL(base,location.href));
    const target=book.path.relative(resolved.pathname)+resolved.hash;
    if(!book.spine.get(target.split("#")[0]))return;
    event.preventDefault();event.stopImmediatePropagation();
    void jumpToBookTarget(target,false,index);
  },true);
  doc.addEventListener("pointerdown",()=>{
    if(snapshot.session.activePane!==index)setActivePane(index);
    closeDrawer(); closeBookNavigation(); setReadingSettings(false);
    workspace.showPanel(false,false);setToolsVisible(false);
  });
  doc.addEventListener("keydown",event=>{
    const target = event.target as HTMLElement;
    if (target?.closest("input, textarea, select, [contenteditable='true']")) return;
    if(event.shiftKey&&['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(event.key))return;
    const table=target.closest<HTMLTableElement>('table');
    if(table&&runtimes[index].readingMode==='scroll'&&table.scrollWidth>table.clientWidth+2&&['ArrowLeft','ArrowRight'].includes(event.key)){event.preventDefault();table.scrollLeft+=event.key==='ArrowLeft'?-100:100;return;}
    if(target.tagName==='IMG'&&!target.closest('a')&&(event.key==='Enter'||event.key===' ')){event.preventDefault();showFigureViewer(target as HTMLImageElement);return;}
    const expandable=target.closest("[data-reader-detail]");
    if(expandable&&(event.key==="Enter"||event.key===" ")) {event.preventDefault();showTextDetail(expandable,index);return;}
    const relevant = ["ArrowLeft","ArrowRight","PageUp","PageDown"," ","Escape","F8","F11","g"].includes(event.key) || event.ctrlKey && (["f","l","n","t"].includes(event.key.toLowerCase())||/^[1-9]$/.test(event.key));
    if (!relevant) return;
    if(snapshot.session.activePane!==index)setActivePane(index);
    const forwarded = new KeyboardEvent("keydown",{key:event.key,code:event.code,ctrlKey:event.ctrlKey,shiftKey:event.shiftKey,altKey:event.altKey,cancelable:true});
    window.dispatchEvent(forwarded);
    if(forwarded.defaultPrevented)event.preventDefault();
  });
  doc.addEventListener("wheel",event=>{
    if(runtimes[index].readingMode==='scroll'){runtimes[index].preserveSemanticFocus=false;if(snapshot.session.activePane!==index)setActivePane(index);return;}
    if(event.ctrlKey||annotationTool!=="read"||doc.getSelection()?.toString())return;
    const state=wheelState[index],now=performance.now();
    const delta=(Math.abs(event.deltaX)>Math.abs(event.deltaY)?event.deltaX:event.deltaY)*(event.deltaMode===1?18:event.deltaMode===2?400:1);
    if(Math.abs(delta)<0.5)return;
    event.preventDefault();
    if(now-state.last>180||Math.sign(delta)!==Math.sign(state.sum))state.sum=0;
    state.last=now; state.sum+=delta;
    if(Math.abs(state.sum)>=55 && now-state.turned>190) { const direction=state.sum>0?"next":"prev";state.sum=0;state.turned=now;if(snapshot.session.activePane!==index){setActivePane(index);contents.window.focus();}void navigate(index,direction); }
  },{passive:false});
  doc.addEventListener("click",event=>{
    const target=event.target as HTMLElement;
    if(doc.getSelection()?.toString())return;
    if(target.tagName === "IMG" && !target.closest("a")) {event.preventDefault();showFigureViewer(target as HTMLImageElement);return;}
    const expandable=target.closest("[data-reader-detail]");
    if(expandable&&!target.closest("a")){event.preventDefault();showTextDetail(expandable,index);}
  });
}
function wireReadingNavigation(): void {
  app.querySelector('.reading-mode')?.addEventListener('change',event=>{void setReadingMode(snapshot.session.activePane,(event.target as HTMLSelectElement).value==='scroll'?'scroll':'paged');});
  app.querySelector('.reader-line-height')?.addEventListener('change',event=>changeReadingPreference('lineHeight',Number((event.target as HTMLInputElement).value)));
  app.querySelector('.reader-content-width')?.addEventListener('change',event=>changeReadingPreference('contentWidth',Number((event.target as HTMLSelectElement).value)));
  app.querySelector('.settings-page-mode')?.addEventListener('change',event=>setBookPageMode(snapshot.session.activePane,Number((event.target as HTMLSelectElement).value)));
  app.querySelector(".reading-back")?.addEventListener("click",returnToPreviousPosition);
  app.querySelector(".contents-toggle")?.addEventListener("click",()=>navigationPanel.classList.contains("visible")&&navigationTab==="contents"?closeBookNavigation():openBookNavigation("contents"));
  app.querySelector(".book-search-toggle")?.addEventListener("click",()=>openBookNavigation("search"));
  app.querySelector(".navigation-close")?.addEventListener("click",closeBookNavigation);
  app.querySelector(".reading-settings-toggle")?.addEventListener("click",()=>setReadingSettings(!settingsPanel.classList.contains("visible")));
  app.querySelector(".settings-close")?.addEventListener("click",()=>setReadingSettings(false));
  app.querySelector(".book-search-form")?.addEventListener("submit",event=>{event.preventDefault();void searchCurrentBook();});
  navigationPanel.addEventListener("click",event=>{
    const target=event.target as HTMLElement;
    const tab=target.closest<HTMLElement>("[data-navigation-tab]")?.dataset.navigationTab;
    if(tab==="contents"||tab==="search")openBookNavigation(tab);
    const href=target.closest<HTMLElement>("[data-chapter-href]")?.dataset.chapterHref;
    if(href)void jumpToBookTarget(href);
    const hit=target.closest<HTMLElement>("[data-search-hit]")?.dataset.searchHit;
    if(hit!==undefined&&searchHits[Number(hit)])void jumpToBookTarget(searchHits[Number(hit)].cfi,true);
  });
  app.querySelector(".reader-font")?.addEventListener("change",event=>{
    changeReadingPreference('readerFont',(event.target as HTMLSelectElement).value as ReaderSession['readerFont']);
  });
  document.addEventListener("pointerdown",event=>{
    const target=event.target as HTMLElement;
    if(!target.closest(".library-drawer,.library-toggle,.left-library-handle,.workspace-library"))closeDrawer();
    if(!target.closest(".book-navigation,.contents-toggle,.book-search-toggle"))closeBookNavigation();
    if(!target.closest(".reading-settings,.reading-settings-toggle"))setReadingSettings(false);
  });
}

type ImportedDraft = { activityId: string; code: string; history: Array<{ code: string; sha256: string; updated_at: number }> };
type PreparedPersonalImport = {
  annotations: ReaderAnnotation[];
  notes: Array<Record<string, any>>;
  archivedNotes: Array<Record<string, any>>;
  unfinished: Record<string, any> | null;
  unfinishedQueue: Array<Record<string, any>>;
  runs: any[];
  drafts: ImportedDraft[];
  restoreCfi: string | null;
};

function personalRecord(value: unknown): value is Record<string, any> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

async function codeDigest(code: string): Promise<string> {
  const bytes = new TextEncoder().encode(code);
  const hash = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return Array.from(hash, byte => byte.toString(16).padStart(2, "0")).join("");
}

async function preparePersonalImport(value: any, restorePosition: boolean): Promise<PreparedPersonalImport> {
  if (!personalRecord(value.progress) || !personalRecord(value.study)) throw new Error("阅读位置或学习记录格式无效；本次未更改数据");
  if (new TextEncoder().encode(JSON.stringify(value)).length > 10 * 1024 * 1024) throw new Error("这份记录超过 10 MB 导入范围；请分批迁移，本次未更改数据");
  const list = (source: unknown, label: string): any[] => {
    if (source === undefined || source === null) return [];
    if (!Array.isArray(source)) throw new Error(`${label}格式无效；本次未更改数据`);
    return source;
  };
  const rawAnnotations = list(value.progress.annotations, "批注");
  const rawNotes = list(value.study.notes, "笔记");
  const rawArchived = list(value.study.archived_notes, "回收站笔记");
  const rawQueue = list(value.study.unfinished_notes, "未提交笔记队列");
  const runs = list(value.runs, "运行收据");
  const rawDrafts = list(value.drafts, "代码草稿");
  if (rawAnnotations.length > 10000 || rawNotes.length + rawArchived.length + rawQueue.length > 10000 || runs.length > 10000 || rawDrafts.length > 500) {
    throw new Error("批注、笔记、收据或活动数量超过本次导入范围；请分批迁移，本次未更改数据");
  }
  for (const annotation of rawAnnotations) {
    if (!personalRecord(annotation) || typeof annotation.id !== "string" || !annotation.id || annotation.id.length > 120 ||
        (annotation.kind === "text-mark" && (typeof annotation.cfiRange !== "string" ||
          (typeof annotation.selectedText === "string" && annotation.selectedText.length > 12000) ||
          (typeof annotation.comment === "string" && annotation.comment.length > 12000))) ||
        (annotation.kind === "free-text" && typeof annotation.text === "string" && annotation.text.length > 24000) ||
        (annotation.kind === "drawing" && (!Array.isArray(annotation.points) || annotation.points.length > 8000 ||
          annotation.points.some((point: unknown) => !personalRecord(point) || !Number.isFinite(point.x) || !Number.isFinite(point.y)))) ||
        !["text-mark", "free-text", "drawing"].includes(annotation.kind)) {
      throw new Error("导入批注的身份或结构无效；本次未更改数据");
    }
  }
  const annotations = normalizeAnnotations(rawAnnotations);
  if (annotations.length !== rawAnnotations.length) throw new Error("导入批注含无法保留的内容；本次未更改数据");
  const validateNote = (note: unknown): note is Record<string, any> =>
    personalRecord(note) && typeof note.id === "string" && note.id.length > 0 && note.id.length <= 120 && typeof note.text === "string";
  if (![...rawNotes, ...rawArchived].every(validateNote)) throw new Error("导入笔记的身份或正文无效；本次未更改数据");
  const activeIds = new Map<string, string>();
  for (const note of rawNotes) {
    const body = JSON.stringify(note);
    const previous = activeIds.get(note.id);
    if (previous && previous !== body) throw new Error("导入文件有重复且不同的活动笔记身份；本次未更改数据");
    activeIds.set(note.id, body);
  }
  const validateUnfinished = async (note: unknown): Promise<Record<string, any>> => {
    if (!personalRecord(note) || typeof note.text !== "string" ||
        (note.editId != null && (typeof note.editId !== "string" || note.editId.length > 120))) {
      throw new Error("未提交笔记格式无效；本次未更改数据");
    }
    const expected = await unfinishedNoteIdentity(note);
    if (note.import_id !== undefined && note.import_id !== expected) throw new Error("未提交笔记的内容身份校验失败；本次未更改数据");
    return note;
  };
  const unfinished = value.study.unfinished_note == null ? null : await validateUnfinished(value.study.unfinished_note);
  const unfinishedQueue: Array<Record<string, any>> = [];
  for (const note of rawQueue) unfinishedQueue.push(await validateUnfinished(note));
  for (const receipt of runs) {
    if (!personalRecord(receipt) || !personalRecord(receipt.record) || typeof receipt.record.run_id !== "string" || !receipt.record.run_id) {
      throw new Error("运行收据结构无效；本次未更改数据");
    }
  }
  const drafts: ImportedDraft[] = [];
  const activities = new Set<string>();
  let codeBytes = 0, versionCount = 0;
  const verifyCode = async (entry: any, label: string): Promise<void> => {
    if (!personalRecord(entry) || typeof entry.code !== "string") throw new Error(`${label}内容无效；本次未更改数据`);
    const size = new TextEncoder().encode(entry.code).length;
    if (size > 300000) throw new Error(`${label}超过单版 300 KB 导入范围；本次未更改数据`);
    codeBytes += size;
    versionCount++;
    if (codeBytes > 10 * 1024 * 1024 || versionCount > 2000) throw new Error("代码历史超过 10 MB 或 2000 版导入范围；请分批迁移，本次未更改数据");
    if (entry.sha256 !== undefined && (typeof entry.sha256 !== "string" || !/^[0-9a-f]{64}$/i.test(entry.sha256) || entry.sha256.toLowerCase() !== await codeDigest(entry.code))) {
      throw new Error(`${label}的 SHA-256 与代码不符；本次未更改数据`);
    }
  };
  for (const draft of rawDrafts) {
    if (!personalRecord(draft) || typeof draft.activityId !== "string" || !/^[a-z][a-z0-9-]{0,63}$/.test(draft.activityId) || activities.has(draft.activityId)) {
      throw new Error("导入草稿的活动身份无效或重复；本次未更改数据");
    }
    activities.add(draft.activityId);
    await verifyCode(draft, "当前草稿");
    const history = list(draft.history, "草稿历史");
    for (const version of history) {
      await verifyCode(version, "草稿历史");
      if (typeof version.sha256 !== "string" || typeof version.updated_at !== "number" || !Number.isFinite(version.updated_at) || version.updated_at < 0) {
        throw new Error("草稿历史缺少有效的哈希或保存时间；本次未更改数据");
      }
    }
    drafts.push({ activityId: draft.activityId, code: draft.code, history });
  }
  let restoreCfi: string | null = null;
  if (restorePosition) {
    const cfi = value.progress.cfi;
    if (typeof cfi !== "string" || !cfi.startsWith("epubcfi(") || cfi.length > 1000) throw new Error("导出文件没有可用的阅读位置；本次未更改数据");
    restoreCfi = cfi;
    if (value.progress.readingMode != null && !["paged", "scroll"].includes(value.progress.readingMode)) throw new Error("阅读方式无效；本次未更改数据");
  }
  return { annotations, notes: rawNotes, archivedNotes: rawArchived, unfinished, unfinishedQueue, runs, drafts, restoreCfi };
}

async function start(): Promise<void> {
  buildPaneShells();
  workspace=new ReadingWorkspace(readerGrid,app,{
    changed:state=>{snapshot.session.workspace=state;persistSession();},
    activate:setActivePane,
    visible:panes=>{queueMicrotask(()=>{if(!readerReady)return;for(const index of panes){const id=snapshot.session.paneBookIds[index];if(id&&!runtimes[index].opening&&!runtimes[index].bookId)void openBook(id,index,undefined,false);}});},
    title:index=>getBook(snapshot.session.paneBookIds[index])?.title??'阅读区域',
    close:index=>{void closeBook(index);},toast:showToast,
  });
  wireEvents();
  await initCatalogue({
    known:uuid=>{const record=snapshot.books.find(book=>book.bookUuid===uuid);return record?{record,sourceHash:snapshot.progress[record.id]?.sourceSha256}:null;},
    openExisting:async(id,chapter)=>{closeDrawer();await openBookInNewPane(id,chapter);},
    current:()=>{const r=runtimes[snapshot.session.activePane];if(!r.bookId)return null;return{id:r.bookId,title:getBook(r.bookId)?.title,href:r.book?.spine.get(r.cfi??0)?.href??''};},
    open:async(record,chapter,replaceExisting)=>{
      const saved=await invoke<CatalogRecord>('register_catalog_book',{record,replaceExisting:replaceExisting??false});
      const old=snapshot.books.findIndex(b=>b.id===saved.id);if(old<0)snapshot.books.push(saved);else snapshot.books[old]=saved;
      renderLibrary();closeDrawer();await openBookInNewPane(saved.id,chapter);
      if(!isDesktop){const url=new URL(location.href);url.searchParams.set('book',record.catalogSource.slug);if(chapter)url.searchParams.set('chapter',chapter);else url.searchParams.delete('chapter');history.replaceState(null,'',url);}
    },
    toast:message=>showToast(message),
    livePersonal:bookId=>{const index=runtimes.findIndex(r=>r.bookId===bookId);return{...learningMemoryBackup(bookId),progress:index>=0?makePaneProgress(index)?.progress??null:null};},
    flushPersonal:async()=>{
      await flushLearningBeforeClose();const index=snapshot.session.activePane;
      if(runtimes[index].opening)await runtimes[index].opening;
      await enqueuePaneOperation(index,async()=>{const value=makePaneProgress(index);if(value)await writePaneProgress(index,value);else await progressWrites[index];});
    },
    importPersonal:async(value,restorePosition)=>{
      const runtime=runtimes[snapshot.session.activePane],record=getBook(runtime.bookId);if(!record||!runtime.bookId)throw new Error('请先打开目标书籍');
      if(value?.format!=='comfortable-reader-personal@1'||!record.bookUuid||record.bookUuid!==value.book?.bookUuid)throw new Error('记录属于另一部书。未修改当前书籍或个人数据。');
      const matchingContent=typeof value.progress?.contentDigest==='string'&&value.progress.contentDigest.length>0&&value.progress.contentDigest===contentDigests.get(runtime.bookId);
      const matchingLocalBytes=!record.catalogSource&&typeof value.progress?.sourceSha256==='string'&&value.progress.sourceSha256.length>0&&value.progress.sourceSha256===sourceDigests.get(runtime.bookId);
      if(!matchingContent&&!matchingLocalBytes)throw new Error('内容身份尚未匹配。旧记录仍完整保留；请从新版阅读器导出同一内容版本，再导入，不能把旧批注直接贴到未经核对的正文。');
      const prepared=await preparePersonalImport(value,restorePosition);
      await closeLearning();
      if(learningIsOpen())throw new Error('学习面板仍有未保存的编辑；本次未导入，请先保存后重试');
      const identity=async(item:unknown)=>'import-'+await codeDigest(JSON.stringify(item));
      const existingProgress=snapshot.progress[runtime.bookId];
      const current=structuredClone(existingProgress??{
        cfi:runtime.cfi,page:runtime.currentPage,totalPages:runtime.totalPages,percent:runtime.percent,
        updatedAt:Math.floor(Date.now()/1000),pageMode:runtime.pageMode,annotations:[],
      } as BookProgress);
      current.annotations=normalizeAnnotations(current.annotations);
      for(const source of prepared.annotations){
        const note=structuredClone(source);
        const old=current.annotations.find(n=>n.id===note.id);
        if(old&&JSON.stringify(old)===JSON.stringify(note))continue;
        if(old){let attempt=0;do{note.id=await identity({kind:'annotation',source,attempt:attempt++});}while(current.annotations.some(n=>n.id===note.id&&JSON.stringify(n)!==JSON.stringify(note)));}
        if(!current.annotations.some(n=>n.id===note.id))current.annotations.push(note);
      }
      if(current.annotations.length>10000)throw new Error('合并后的批注超过 10000 条；请分批迁移，本次未更改数据');
      const state=await invoke<any>('learning_load_state',{bookId:runtime.bookId});
      if(!personalRecord(state)||!Array.isArray(state.notes)||
          (state.archived_notes!=null&&!Array.isArray(state.archived_notes))||
          (state.unfinished_notes!=null&&!Array.isArray(state.unfinished_notes))||
          (state.importedRuns!=null&&!Array.isArray(state.importedRuns)))throw new Error('本机学习记录结构异常；本次未更改数据');
      state.archived_notes??=[];state.unfinished_notes??=[];state.importedRuns??=[];
      const importedNoteIds=new Map<string,string>();
      const mergeNote=async(source:Record<string,any>,bucket:'notes'|'archived_notes'):Promise<string>=>{
        const note=structuredClone(source);
        if(note.run_id){note.imported_run_id=note.run_id;delete note.run_id;}
        const target:Record<string,any>[]=state[bucket];
        const exact=target.find(row=>row.id===note.id&&JSON.stringify(row)===JSON.stringify(note));
        if(exact)return exact.id;
        const originalId=note.id;
        if([...state.notes,...state.archived_notes].some((row:any)=>row.id===originalId)){
          let attempt=0;
          do{note.id=await identity({kind:bucket,source:note,attempt:attempt++});}
          while([...state.notes,...state.archived_notes].some((row:any)=>row.id===note.id&&JSON.stringify(row)!==JSON.stringify(note)));
        }
        if(!target.some(row=>row.id===note.id))target.push(note);
        return note.id;
      };
      for(const note of prepared.notes)importedNoteIds.set(note.id,await mergeNote(note,'notes'));
      for(const note of prepared.archivedNotes)await mergeNote(note,'archived_notes');
      const currentUnfinished=state.unfinished_note;
      if(currentUnfinished!=null&&(!personalRecord(currentUnfinished)||typeof currentUnfinished.text!=='string'))throw new Error('本机未提交笔记结构异常；本次未更改数据');
      const queuedIds=new Set<string>();
      for(const queued of state.unfinished_notes){
        if(!personalRecord(queued)||typeof queued.text!=='string')throw new Error('本机未提交笔记队列结构异常；本次未更改数据');
        queued.import_id=await unfinishedNoteIdentity(queued);
        queuedIds.add(queued.import_id);
      }
      let activeUnfinishedId=currentUnfinished?await unfinishedNoteIdentity(currentUnfinished):null;
      const mergeUnfinished=async(source:Record<string,any>,makeCurrent:boolean):Promise<void>=>{
        const note=structuredClone(source);delete note.import_id;
        if(note.editId!=null)note.editId=importedNoteIds.get(note.editId)??null;
        const importId=await unfinishedNoteIdentity(note);
        if(importId===activeUnfinishedId||queuedIds.has(importId))return;
        if(makeCurrent&&!state.unfinished_note){state.unfinished_note=note;activeUnfinishedId=importId;return;}
        state.unfinished_notes.push({...note,import_id:importId});queuedIds.add(importId);
      };
      if(prepared.unfinished)await mergeUnfinished(prepared.unfinished,true);
      for(const note of prepared.unfinishedQueue)await mergeUnfinished(note,false);
      if(state.notes.length+state.archived_notes.length+state.unfinished_notes.length>10000)throw new Error('合并后的笔记超过 10000 条；请分批迁移，本次未更改数据');
      for(const receipt of prepared.runs){
        const old=state.importedRuns.find((r:any)=>r?.record?.run_id===receipt.record.run_id);
        if(old&&JSON.stringify(old)!==JSON.stringify(receipt))throw new Error('运行收据身份冲突；本次未更改数据');
        if(!old)state.importedRuns.push(receipt);
      }
      if(state.importedRuns.length>10000)throw new Error('合并后的运行收据超过 10000 条；请分批迁移，本次未更改数据');
      if(new TextEncoder().encode(JSON.stringify(state)).length>4*1024*1024)throw new Error('合并后的学习记录超过 4 MB，请分批迁移；本次未更改数据');
      let committedProgress=false,committedStudy=false,committedDrafts=0;
      try{
        await invoke('save_progress',{bookId:runtime.bookId,progress:current});snapshot.progress[runtime.bookId]=current;committedProgress=true;
        await invoke('learning_save_state',{bookId:runtime.bookId,value:state});committedStudy=true;
        for(const draft of prepared.drafts){
          await invoke('learning_import_draft',{bookId:runtime.bookId,activityId:draft.activityId,code:draft.code});committedDrafts++;
          for(const version of draft.history){await invoke('learning_import_draft',{bookId:runtime.bookId,activityId:draft.activityId,code:version.code});committedDrafts++;}
        }
      }catch(error){
        applyTextAnnotations(snapshot.session.activePane);renderOverlayAnnotations(snapshot.session.activePane);renderAnnotationPanel();
        const done=[committedProgress?'书页批注':'',committedStudy?'学习笔记':'',committedDrafts?`${committedDrafts} 份代码历史`:''].filter(Boolean).join('、');
        throw new Error(`${done?'合并只完成了一部分：已保存'+done+'；其余尚未保存。':'合并尚未保存。'}旧记录没有被删除。请保留导入文件，解决保存问题后重试；相同记录会去重。具体原因：${readableError(error)}`);
      }
      if(prepared.restoreCfi){if(!await restoreImportedPosition(prepared.restoreCfi,snapshot.session.activePane))throw new Error('记录已合并，但目标内容尚未取得。原阅读位置保留；联网或准备所需章节后可再次恢复。');if(['paged','scroll'].includes(value.progress.readingMode))await setReadingMode(snapshot.session.activePane,value.progress.readingMode);}
      applyTextAnnotations(snapshot.session.activePane);renderOverlayAnnotations(snapshot.session.activePane);renderAnnotationPanel();showToast('已合并批注、笔记和代码历史；已有当前草稿保留。运行收据只读，不会自动执行。');
    },
  });
  initLearning({
    current: () => {
      const pane = snapshot.session.activePane;
      const r = runtimes[pane];
      if (!r.bookId || !r.book) return null;
      const section = r.book.spine.get(r.cfi ?? 0);
      return { bookId: r.bookId, contentDigest:contentDigests.get(r.bookId),sourceSha256:sourceDigests.get(r.bookId),localSource:Boolean((r.book as any).archive),pane, cfi: r.cfi, href: section ? section.href : "", quote: "", focus: document.activeElement as HTMLElement | null };
    },
    settle:()=>enqueuePaneOperation(snapshot.session.activePane,async()=>{}),
    jump: (target, pane) => jumpToBookTarget(target, false, pane),
    toast: (text) => showToast(text, "error"),
  });
  await getCurrentWindow().onCloseRequested(async (event) => {
    if(!readerReady)return;
    try {
      await flushLearningBeforeClose();
      await Promise.all(progressWrites);
      if (sessionSaveTimer !== null) window.clearTimeout(sessionSaveTimer);
      for (let index=0;index<MAX_PANES;index++) {
        if(progressSaveTimers[index]!==null)window.clearTimeout(progressSaveTimers[index]!);
        const runtime=runtimes[index];if(!runtime.bookId||editionHeld(runtime.bookId)||runtime.restoringLocation||(!runtime.cfi&&!runtime.lazyPageCount))continue;
        const existing=snapshot.progress[runtime.bookId];
        await invoke("save_progress",{bookId:runtime.bookId,progress:{...existing,sourceSha256:sourceDigests.get(runtime.bookId)??existing?.sourceSha256??null,cfi:runtime.lazyPageCount>0?null:runtime.cfi??existing?.cfi??null,page:runtime.currentPage||existing?.page||0,totalPages:runtime.totalPages||existing?.totalPages||0,percent:runtime.cfi?runtime.percent:existing?.percent??0,pageMode:runtime.pageMode,annotations:existing?.annotations??[],updatedAt:Math.floor(Date.now()/1000)}});
      }
      await invoke("save_session",{session:sessionPayload()});
    } catch(error) {
      event.preventDefault();showToast(`保存未完成，窗口保持打开：${String(error)}`,"error");
    }
  });
  try {
    const loaded = await invoke<AppSnapshot>("bootstrap");
    snapshot = {
      ...loaded,
      progress: normalizeProgress(loaded.progress),
      session: normalizeSession(loaded.session),
    };
    snapshot.session.paneBookIds = snapshot.session.paneBookIds.map((bookId) =>
      getBook(bookId) ? bookId : null,
    );
    if (!snapshot.session.paneBookIds.some(Boolean) && !snapshot.session.workspace && snapshot.books[0]) {
      snapshot.session.paneBookIds[0] = snapshot.books[0].id;
    }
    snapshot.session.activePane = Math.min(
      snapshot.session.activePane,
      snapshot.session.paneCount - 1,
    );
    if(!snapshot.session.paneBookIds[snapshot.session.activePane])snapshot.session.activePane=Math.max(0,snapshot.session.paneBookIds.findIndex(Boolean));
    if(!isDesktop){
      try { await openWebLink(); }
      catch(error) { showToast(`这条书籍链接暂时打不开：${readableError(error)}。可在“发现书籍”中重试。`,"error"); }
    }
    renderLibrary();
    updateLayoutUi();
    applyTheme();
    applyFontScale();
    persistSession();

    await Promise.allSettled(
      workspace.visiblePanes.map(index=>{const bookId=snapshot.session.paneBookIds[index];return bookId?openBook(bookId,index,undefined,false):Promise.resolve();}),
    );
    readerReady=true;
    if(!isDesktop&&'serviceWorker' in navigator)void navigator.serviceWorker.register(new URL('sw.js',location.href)).catch(()=>showToast('离线页面尚未准备，已保存的材料仍在。'));
    window.setTimeout(() => bootScreen.classList.add("hidden"), 220);
    if(!isDesktop&&!snapshot.session.paneBookIds.some(Boolean))void showCatalogue();
  } catch (error) {
    bootScreen.innerHTML = `
      <div class="boot-mark error">${icons.close}</div>
      <strong>书库没有成功启动</strong>
      <span>${escapeHtml(String(error))}</span>`;
  }
}

void start();
