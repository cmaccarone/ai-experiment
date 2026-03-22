import { describe, it, expect } from 'vitest';
import { buildFontRuns } from './pdf-renderer.js';
import type { MeasuredWord } from './types.js';

function word(text: string, width: number, fontStyle = 'body' as const): MeasuredWord {
  return { text, width, fontStyle };
}

function space(width = 5): MeasuredWord {
  return { text: ' ', width, fontStyle: 'body' };
}

describe('buildFontRuns', () => {
  it('returns empty array for no words', () => {
    expect(buildFontRuns([])).toEqual([]);
  });

  it('returns empty array for only whitespace words', () => {
    expect(buildFontRuns([space(), space()])).toEqual([]);
  });

  it('groups a single word into one run', () => {
    const runs = buildFontRuns([word('hello', 30)]);
    expect(runs).toEqual([
      { text: 'hello', fontStyle: 'body', contentWidth: 30, wordCount: 1 },
    ]);
  });

  it('groups consecutive same-font words with spaces between them', () => {
    const runs = buildFontRuns([
      word('hello', 30),
      space(),
      word('world', 32),
    ]);
    expect(runs).toHaveLength(1);
    expect(runs[0].text).toBe('hello world');
    expect(runs[0].contentWidth).toBe(62);
    expect(runs[0].wordCount).toBe(2);
  });

  it('splits into separate runs when font style changes', () => {
    const runs = buildFontRuns([
      word('normal', 40),
      space(),
      { text: 'italic', width: 35, fontStyle: 'bodyItalic' as const },
      space(),
      word('back', 25),
    ]);
    expect(runs).toHaveLength(3);
    expect(runs[0]).toEqual({ text: 'normal', fontStyle: 'body', contentWidth: 40, wordCount: 1 });
    expect(runs[1]).toEqual({ text: 'italic', fontStyle: 'bodyItalic', contentWidth: 35, wordCount: 1 });
    expect(runs[2]).toEqual({ text: 'back', fontStyle: 'body', contentWidth: 25, wordCount: 1 });
  });

  it('groups multiple same-font words across whitespace tokens', () => {
    const runs = buildFontRuns([
      word('the', 15),
      space(),
      word('quick', 30),
      space(),
      word('brown', 32),
      space(),
      word('fox', 18),
    ]);
    expect(runs).toHaveLength(1);
    expect(runs[0].text).toBe('the quick brown fox');
    expect(runs[0].contentWidth).toBe(95);
    expect(runs[0].wordCount).toBe(4);
  });

  it('handles adjacent content words without whitespace between them', () => {
    const runs = buildFontRuns([
      word('hello', 30),
      word('world', 32),
    ]);
    expect(runs).toHaveLength(1);
    expect(runs[0].text).toBe('hello world');
    expect(runs[0].wordCount).toBe(2);
  });

  it('handles mixed fonts with multiple words per run', () => {
    const runs = buildFontRuns([
      word('this', 20),
      space(),
      word('is', 10),
      space(),
      { text: 'very', width: 22, fontStyle: 'bodyBold' as const },
      space(),
      { text: 'important', width: 50, fontStyle: 'bodyBold' as const },
      space(),
      word('text', 22),
    ]);
    expect(runs).toHaveLength(3);
    expect(runs[0].text).toBe('this is');
    expect(runs[0].wordCount).toBe(2);
    expect(runs[1].text).toBe('very important');
    expect(runs[1].fontStyle).toBe('bodyBold');
    expect(runs[1].wordCount).toBe(2);
    expect(runs[2].text).toBe('text');
    expect(runs[2].wordCount).toBe(1);
  });

  it('preserves space characters in run text for PDF extraction', () => {
    const runs = buildFontRuns([
      word('descendants', 60),
      space(),
      word('inherited', 50),
      space(),
      word('not', 18),
      space(),
      word('only', 22),
    ]);
    // All same font => single run with spaces embedded
    expect(runs).toHaveLength(1);
    expect(runs[0].text).toBe('descendants inherited not only');
    // Space characters are in the string, ensuring PDF extractors see them
    expect(runs[0].text).toContain(' ');
    expect(runs[0].text.split(' ')).toHaveLength(4);
  });
});
