// epub.js 0.3.93 emits these errors without rejecting its pending public
// promises. Settle the original deferred so its queue and our recovery UI agree.
export class ReaderContentLoadError extends Error {
  constructor(cause: unknown) {
    super(cause instanceof Error ? cause.message : String(cause));
    this.name = 'ReaderContentLoadError';
  }
}

type BookWithFailure = {
  on: (name: string, listener: (error: unknown) => void) => void;
  opening: { reject: (error: unknown) => void };
};
type RenditionWithFailure = {
  on: (name: string, listener: (error: unknown) => void) => void;
  displaying?: { reject: (error: unknown) => void };
};

export function propagateBookOpenFailure(book: BookWithFailure): void {
  book.on('openFailed', error => book.opening.reject(new ReaderContentLoadError(error)));
}

export function propagateDisplayFailure(rendition: RenditionWithFailure): void {
  // The implementation emits lowercase displayerror; its JSDoc spells it
  // displayError. Follow the runtime constant, not the comment.
  rendition.on('displayerror', error => {
    const pending = rendition.displaying;
    rendition.displaying = undefined;
    pending?.reject(new ReaderContentLoadError(error));
  });
}

export function readableError(error: unknown): string {
  return (error instanceof Error ? error.message : String(error))
    .replace(/^(?:Error|ReaderContentLoadError):\s*/i, '').replace(/[。.!\s]+$/, '');
}
