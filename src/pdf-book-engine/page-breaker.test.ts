import { describe, it, expect } from 'vitest';
import { breakPages, type ChapterContent, type PageItem } from './page-breaker.js';

function makeLine(height: number = 14): PageItem {
  return {
    type: 'line',
    height,
    typesetLine: { words: [], width: 0, availableWidth: 300, isLastLine: false },
  };
}

function makeChapter(lineCount: number, title: string = 'Chapter', forceRecto: boolean = true): ChapterContent {
  const items: PageItem[] = [];
  for (let i = 0; i < lineCount; i++) {
    items.push(makeLine(14));
  }
  return { title, forceRecto, items };
}

describe('breakPages', () => {
  it('puts content on a single page when it fits', () => {
    const chapters = [makeChapter(5)];
    const pages = breakPages(chapters, { availableHeight: 200, widowLines: 2, orphanLines: 2 });
    expect(pages).toHaveLength(1);
    expect(pages[0].items).toHaveLength(5);
    expect(pages[0].isChapterOpener).toBe(true);
  });

  it('breaks across multiple pages', () => {
    // 50 lines * 14pt = 700pt, with 200pt pages should need 4 pages
    const chapters = [makeChapter(50)];
    const pages = breakPages(chapters, { availableHeight: 200, widowLines: 2, orphanLines: 2 });
    expect(pages.length).toBeGreaterThan(1);
    expect(pages[0].isChapterOpener).toBe(true);
    for (let i = 1; i < pages.length; i++) {
      expect(pages[i].isChapterOpener).toBe(false);
    }
  });

  it('starts a new chapter on a new page', () => {
    const chapters = [
      makeChapter(5, 'Chapter 1'),
      makeChapter(5, 'Chapter 2'),
    ];
    const pages = breakPages(chapters, { availableHeight: 200, widowLines: 2, orphanLines: 2 });
    // Should be at least 2 pages (one per chapter), possibly 3 with recto padding
    expect(pages.length).toBeGreaterThanOrEqual(2);
  });

  it('adds blank page for recto start when needed', () => {
    // Chapter 1 takes 1 page (odd = recto), Chapter 2 needs a blank verso before it
    const chapters = [
      makeChapter(3, 'Ch1', true),
      makeChapter(3, 'Ch2', true),
    ];
    const pages = breakPages(chapters, { availableHeight: 200, widowLines: 2, orphanLines: 2 });
    // Page 1: Ch1 (recto), Page 2: blank (verso), Page 3: Ch2 (recto)
    expect(pages.length).toBeGreaterThanOrEqual(3);
    expect(pages[1].items).toHaveLength(0); // blank page
    expect(pages[2].isChapterOpener).toBe(true);
  });

  it('records chapter title on each page', () => {
    const chapters = [makeChapter(5, 'My Chapter')];
    const pages = breakPages(chapters, { availableHeight: 200, widowLines: 2, orphanLines: 2 });
    expect(pages[0].chapterTitle).toBe('My Chapter');
  });
});
