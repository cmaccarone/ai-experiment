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
