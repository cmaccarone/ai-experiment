import { describe, it, expect } from 'vitest';
import { generateTypstDocument, escapeTypst } from './typst-generator.js';
import type { Chapter, PdfBookConfig } from './types.js';

function makeConfig(overrides: Partial<PdfBookConfig> = {}): PdfBookConfig {
  return {
    trimWidth: 6,
    trimHeight: 9,
    margins: { top: 0.75, bottom: 0.75, inside: 0.75, outside: 0.5 },
    fonts: { body: '/fonts/body.ttf' },
    fontSize: 11,
    lineHeight: 1.4,
    paragraphIndent: 1.5,
    chapterStartRecto: true,
    chapterTopDrop: 2,
    widowLines: 2,
    orphanLines: 2,
    ...overrides,
  };
}

const simpleChapters: Chapter[] = [
  {
    title: 'First Chapter',
    html: '<p>Hello world.</p>',
  },
];

describe('escapeTypst', () => {
  it('escapes hash characters', () => {
    expect(escapeTypst('#heading')).toBe('\\#heading');
  });

  it('escapes dollar signs', () => {
    expect(escapeTypst('$100')).toBe('\\$100');
  });

  it('escapes angle brackets', () => {
    expect(escapeTypst('a < b > c')).toBe('a \\< b \\> c');
  });

  it('escapes at signs', () => {
    expect(escapeTypst('user@example')).toBe('user\\@example');
  });

  it('escapes backslashes', () => {
    expect(escapeTypst('a\\b')).toBe('a\\\\b');
  });

  it('leaves normal text unchanged', () => {
    expect(escapeTypst('Hello world')).toBe('Hello world');
  });
});

describe('generateTypstDocument', () => {
  it('includes page setup with correct dimensions', () => {
    const result = generateTypstDocument(simpleChapters, makeConfig(), 0.625);
    expect(result).toContain('width: 6in');
    expect(result).toContain('height: 9in');
    expect(result).toContain('inside: 0.625in');
    expect(result).toContain('outside: 0.5in');
    expect(result).toContain('top: 0.75in');
    expect(result).toContain('bottom: 0.75in');
  });

  it('sets font size and paragraph properties', () => {
    const result = generateTypstDocument(simpleChapters, makeConfig(), 0.5);
    expect(result).toContain('size: 11pt');
    expect(result).toContain('justify: true');
    expect(result).toContain('first-line-indent: 1.5em');
  });

  it('generates chapter title', () => {
    const result = generateTypstDocument(simpleChapters, makeConfig(), 0.5);
    expect(result).toContain('First Chapter');
  });

  it('generates chapter top drop', () => {
    const result = generateTypstDocument(simpleChapters, makeConfig({ chapterTopDrop: 2 }), 0.5);
    expect(result).toContain('#v(2in)');
  });

  it('generates recto page break for chapters', () => {
    const chapters: Chapter[] = [
      { title: 'One', html: '<p>Text.</p>' },
      { title: 'Two', html: '<p>More text.</p>' },
    ];
    const result = generateTypstDocument(chapters, makeConfig({ chapterStartRecto: true }), 0.5);
    expect(result).toContain('#pagebreak(to: "odd")');
  });

  it('generates regular page break when chapterStartRecto is false', () => {
    const chapters: Chapter[] = [
      { title: 'One', html: '<p>Text.</p>' },
      { title: 'Two', html: '<p>More text.</p>' },
    ];
    const result = generateTypstDocument(chapters, makeConfig({ chapterStartRecto: false }), 0.5);
    expect(result).toContain('#pagebreak()');
    expect(result).not.toContain('to: "odd"');
  });

  it('converts bold and italic runs', () => {
    const chapters: Chapter[] = [
      { title: 'Test', html: '<p>Normal <strong>bold</strong> and <em>italic</em> text.</p>' },
    ];
    const result = generateTypstDocument(chapters, makeConfig(), 0.5);
    expect(result).toContain('*bold*');
    expect(result).toContain('_italic_');
  });

  it('converts links', () => {
    const chapters: Chapter[] = [
      { title: 'Test', html: '<p>See <a href="https://example.com">this link</a>.</p>' },
    ];
    const result = generateTypstDocument(chapters, makeConfig(), 0.5);
    expect(result).toContain('#link("https://example.com")');
  });

  it('converts unordered lists', () => {
    const chapters: Chapter[] = [
      { title: 'Test', html: '<ul><li>First</li><li>Second</li></ul>' },
    ];
    const result = generateTypstDocument(chapters, makeConfig(), 0.5);
    expect(result).toContain('- First');
    expect(result).toContain('- Second');
  });

  it('converts ordered lists', () => {
    const chapters: Chapter[] = [
      { title: 'Test', html: '<ol><li>First</li><li>Second</li></ol>' },
    ];
    const result = generateTypstDocument(chapters, makeConfig(), 0.5);
    expect(result).toContain('+ First');
    expect(result).toContain('+ Second');
  });

  it('converts headings', () => {
    const chapters: Chapter[] = [
      { title: 'Test', html: '<h2>Section</h2><h3>Subsection</h3>' },
    ];
    const result = generateTypstDocument(chapters, makeConfig(), 0.5);
    expect(result).toContain('== Section');
    expect(result).toContain('=== Subsection');
  });

  it('handles empty chapter HTML', () => {
    const chapters: Chapter[] = [
      { title: 'Empty', html: '' },
    ];
    // Should not throw
    const result = generateTypstDocument(chapters, makeConfig(), 0.5);
    expect(result).toContain('Empty');
  });

  it('includes header/footer when configured', () => {
    const result = generateTypstDocument(simpleChapters, makeConfig({
      header: { outside: 'PAGE', inside: 'CHAPTER', separator: '|' },
      footer: { center: 'PAGE' },
    }), 0.5);
    expect(result).toContain('header: locate');
    expect(result).toContain('footer: locate');
    expect(result).toContain('counter(page)');
  });

  it('resolves PAGE and CHAPTER placeholders in header', () => {
    const result = generateTypstDocument(simpleChapters, makeConfig({
      header: { outside: 'PAGE', inside: 'CHAPTER', separator: '|' },
    }), 0.5);
    expect(result).toContain('#str(n)');
    expect(result).toContain('#chapter-title');
    expect(result).toContain('calc.odd(n)');
  });

  it('hides header on chapter opener pages by default', () => {
    const result = generateTypstDocument(simpleChapters, makeConfig({
      header: { outside: 'PAGE' },
    }), 0.5);
    expect(result).toContain('chapter-pages');
    expect(result).toContain('if n in chapter-pages');
  });

  it('respects hideOnChapterOpener: false', () => {
    const result = generateTypstDocument(simpleChapters, makeConfig({
      header: { outside: 'PAGE', hideOnChapterOpener: false },
    }), 0.5);
    expect(result).not.toContain('chapter-pages');
  });

  it('uses binding: left for recto/verso', () => {
    const result = generateTypstDocument(simpleChapters, makeConfig(), 0.5);
    expect(result).toContain('binding: left');
  });
});
