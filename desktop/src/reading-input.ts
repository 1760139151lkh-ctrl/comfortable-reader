/** One intent per touchpad gesture, discrete wheel notches remain responsive. */
export class WheelPager {
  private sum = 0;
  private last = -Infinity;
  private turned = -Infinity;
  private magnitude = 0;
  private direction = 0;
  private latched = false;
  private continuous = false;

  feed(delta: number, mode: number, now: number): -1 | 0 | 1 {
    if (!Number.isFinite(delta) || Math.abs(delta) < .5) return 0;
    const sign = Math.sign(delta), gap = now - this.last, size = Math.abs(delta);
    const newGesture = gap > 160 || sign !== this.direction;
    if (newGesture) this.continuous = false;
    if (mode === 0 && (size !== 100 && size !== 120 || gap < 90 && this.latched && size !== this.magnitude)) this.continuous = true;
    const discrete = mode !== 0 || !this.continuous;
    // A new deliberate acceleration can follow an inertial tail without a long lockout.
    const renewed = !discrete && this.latched && now - this.turned > 160 && size > Math.max(18, this.magnitude * 2.4);
    if (newGesture || renewed) { this.sum = 0; this.latched = false; }
    this.last = now; this.direction = sign; this.magnitude = size;
    if (discrete) {
      if (now - this.turned < 75 && !newGesture) return 0;
      this.turned = now; this.latched = true; this.sum = 0;
      return sign as -1 | 1;
    }
    if (this.latched) return 0;
    this.sum += delta;
    if (Math.abs(this.sum) < 52) return 0;
    this.sum = 0; this.latched = true; this.turned = now;
    return sign as -1 | 1;
  }
}

/** A scrollable object owns its gesture, even at its boundary. Never turn the book behind it. */
export function ownsWheel(target: EventTarget | null, stop: HTMLElement): boolean {
  let element = target && 'nodeType' in target ? target as HTMLElement : null;
  if (element?.nodeType !== 1) element = element?.parentElement ?? null;
  for (; element && element !== stop; element = element.parentElement) {
    if (element.matches('input,textarea,select,video,audio,[contenteditable="true"],[role="slider"]')) return true;
    const style = element.ownerDocument.defaultView!.getComputedStyle(element);
    if (/(auto|scroll)/.test(style.overflowX) && element.scrollWidth > element.clientWidth + 2) return true;
    if (/(auto|scroll)/.test(style.overflowY) && element.scrollHeight > element.clientHeight + 2) return true;
  }
  return false;
}
