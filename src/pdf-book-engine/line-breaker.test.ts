import { describe, it, expect } from 'vitest';
import { breakLines, breakLinesGreedy } from './line-breaker.js';
import type { MeasuredWord } from './types.js';

function makeWords(texts: string[], charWidth: number = 10): MeasuredWord[] {
  const words: MeasuredWord[] = [];
  for (const text of texts) {
    words.push({ text, width: text.length * charWidth, fontStyle: 'body' });
    // Add space after each word except the last
    if (text !== texts[texts.length - 1]) {
      words.push({ text: ' ', width: charWidth, fontStyle: 'body' });
    }
  }
  return words;
}

describe('breakLines (Knuth-Plass)', () => {
  it('returns empty array for no words', () => {
    expect(breakLines([], 300, 10)).toEqual([]);
  });

  it('puts all words on one line when they fit', () => {
    const words = makeWords(['Hello', 'world'], 10);
    const lines = breakLines(words, 500, 10);
    expect(lines).toHaveLength(1);
    expect(lines[0].isLastLine).toBe(true);
  });

  it('breaks long text into multiple lines', () => {
    const words = makeWords(['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog'], 10);
    const lines = breakLines(words, 120, 10); // ~12 chars per line
    expect(lines.length).toBeGreaterThan(1);
    expect(lines[lines.length - 1].isLastLine).toBe(true);
  });

  it('handles a single long word that exceeds line width', () => {
    const words: MeasuredWord[] = [
      { text: 'superlongword', width: 200, fontStyle: 'body' },
    ];
    const lines = breakLines(words, 100, 10);
    expect(lines.length).toBeGreaterThanOrEqual(1);
  });

  it('sets availableWidth on each line', () => {
    const words = makeWords(['Hello', 'world'], 10);
    const lines = breakLines(words, 300, 10);
    for (const line of lines) {
      expect(line.availableWidth).toBe(300);
    }
  });

  it('preserves adjacency for tokens without whitespace between them', () => {
    // Simulates <em>word</em>, where "word" and "," are adjacent (no space)
    const words: MeasuredWord[] = [
      { text: 'She', width: 30, fontStyle: 'body' },
      { text: ' ', width: 10, fontStyle: 'body' },
      { text: 'said', width: 40, fontStyle: 'body' },
      { text: ' ', width: 10, fontStyle: 'body' },
      { text: 'hello', width: 50, fontStyle: 'bodyItalic' },
      // No whitespace token here — comma is adjacent to "hello"
      { text: ',', width: 5, fontStyle: 'body' },
      { text: ' ', width: 10, fontStyle: 'body' },
      { text: 'then', width: 40, fontStyle: 'body' },
      { text: ' ', width: 10, fontStyle: 'body' },
      { text: 'left', width: 40, fontStyle: 'body' },
    ];
    const lines = breakLines(words, 500, 10);
    expect(lines).toHaveLength(1);

    // The comma should be adjacent to "hello" with no space between them
    const texts = lines[0].words.map(w => w.text);
    const helloIdx = texts.indexOf('hello');
    const commaIdx = texts.indexOf(',');
    expect(commaIdx).toBe(helloIdx + 1); // directly adjacent, no space token between
  });

  it('counts space boundaries correctly for justification width', () => {
    // "word, more" — comma adjacent to "word", space before "more"
    const words: MeasuredWord[] = [
      { text: 'word', width: 40, fontStyle: 'bodyItalic' },
      { text: ',', width: 5, fontStyle: 'body' },
      { text: ' ', width: 10, fontStyle: 'body' },
      { text: 'more', width: 40, fontStyle: 'body' },
    ];
    const lines = breakLines(words, 500, 10);
    expect(lines).toHaveLength(1);

    // Should have exactly one space token (between "," and "more")
    const spaceTokens = lines[0].words.filter(w => /^\s+$/.test(w.text));
    expect(spaceTokens).toHaveLength(1);
  });

  it('uses narrower width for first line when firstLineWidth is provided', () => {
    // 6 words, each 40pts wide, space 10pts
    // firstLineWidth = 100 → fits 2 words (40 + 10 + 40 = 90)
    // availableWidth = 200 → fits 4 words (40+10+40+10+40+10+40 = 190)
    const words = makeWords(['aaa', 'bbb', 'ccc', 'ddd', 'eee', 'fff'], 10);
    // Each word is 30pts (3 chars * 10), space is 10pts
    // firstLineWidth = 80 → fits 2 words (30 + 10 + 30 = 70, fits; +10+30=110, no)
    // availableWidth = 200 → fits many words
    const lines = breakLines(words, 200, 10, 80);

    expect(lines.length).toBeGreaterThan(1);
    // First line should have narrower availableWidth
    expect(lines[0].availableWidth).toBe(80);
    // Subsequent lines should use the full width
    for (let i = 1; i < lines.length; i++) {
      expect(lines[i].availableWidth).toBe(200);
    }
  });

  it('does not produce over-stretched spaces on subsequent lines with firstLineWidth', () => {
    // Regression test: previously, all lines were broken to firstLineWidth but
    // subsequent lines had availableWidth set to the wider width, causing
    // justification to over-stretch spaces.
    const words = makeWords(
      ['The', 'voyage', 'took', 'roughly', 'two', 'weeks', 'and', 'the', 'ship', 'arrived'],
      10,
    );
    const availableWidth = 200;
    const firstLineWidth = 150; // 50pts narrower (paragraph indent)
    const spaceWidth = 10;

    const lines = breakLines(words, availableWidth, spaceWidth, firstLineWidth);

    for (const line of lines) {
      // Content width should not exceed the line's availableWidth
      const contentWidth = line.words
        .filter(w => !/^\s+$/.test(w.text))
        .reduce((sum, w) => sum + w.width, 0);
      const spaceCount = line.words.filter(w => /^\s+$/.test(w.text)).length;

      // With proper breaking, content + natural spaces should be <= availableWidth
      const naturalWidth = contentWidth + spaceCount * spaceWidth;
      expect(naturalWidth).toBeLessThanOrEqual(line.availableWidth + 1); // +1 for float rounding
    }
  });
});

describe('breakLinesGreedy', () => {
  it('returns empty array for no words', () => {
    expect(breakLinesGreedy([], 300)).toEqual([]);
  });

  it('puts all words on one line when they fit', () => {
    const words = makeWords(['Hello', 'world'], 10);
    const lines = breakLinesGreedy(words, 500);
    expect(lines).toHaveLength(1);
  });

  it('breaks into multiple lines when needed', () => {
    const words = makeWords(['aa', 'bb', 'cc', 'dd', 'ee'], 10);
    const lines = breakLinesGreedy(words, 60); // ~6 chars fit
    expect(lines.length).toBeGreaterThan(1);
  });
});
