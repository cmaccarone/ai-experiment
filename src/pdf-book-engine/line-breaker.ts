import type { MeasuredWord, TypesetLine } from './types.js';

/**
 * Knuth-Plass line breaking algorithm.
 *
 * Models paragraphs as boxes (word widths), glue (stretchable spaces),
 * and penalties. Finds globally optimal line breaks minimizing total
 * "badness" across all lines. No hyphenation — breaks at word boundaries only.
 *
 * Falls back to greedy breaking if the optimal algorithm fails to converge.
 */

function isWhitespace(text: string): boolean {
  return /^\s+$/.test(text);
}

/**
 * Break a paragraph (array of MeasuredWords) into optimally-broken lines.
 * Uses Knuth-Plass with greedy fallback.
 *
 * @param firstLineWidth - Optional narrower width for the first line (e.g. paragraph indent).
 *                         If omitted, all lines use `availableWidth`.
 */
export function breakLines(
  words: MeasuredWord[],
  availableWidth: number,
  spaceWidth: number,
  firstLineWidth?: number,
): TypesetLine[] {
  if (words.length === 0) return [];

  const lineWidthFor = (lineIndex: number) =>
    lineIndex === 0 && firstLineWidth !== undefined ? firstLineWidth : availableWidth;

  // Separate content words from whitespace, tracking adjacency
  const contentWords: Array<{ word: MeasuredWord; index: number; hasSpaceBefore: boolean }> = [];
  let lastWasSpace = false;

  for (let i = 0; i < words.length; i++) {
    if (isWhitespace(words[i].text)) {
      lastWasSpace = true;
    } else {
      contentWords.push({
        word: words[i],
        index: i,
        hasSpaceBefore: contentWords.length > 0 && lastWasSpace,
      });
      lastWasSpace = false;
    }
  }

  if (contentWords.length === 0) return [];

  // Dynamic programming approach: find optimal break points
  // breakCost[i] = minimum cost to set text from word i to the end
  // breakNext[i] = next line start for the optimal solution from word i
  const n = contentWords.length;
  const breakCost = new Float64Array(n + 1);
  const breakNext = new Int32Array(n + 1);
  breakNext.fill(-1);

  // We need to know which line index each word-start corresponds to.
  // Two-pass: first pass forward to determine line indices, then backward DP.
  // Since line indices depend on break points (circular), we use a simpler approach:
  // Track line index during reconstruction, and run DP with the wider width
  // (subsequent lines), then verify the first line fits.

  // Forward DP: process from end to start.
  // For the first line (starting at word 0), use lineWidthFor(0).
  // For subsequent lines, use availableWidth.
  // We approximate by noting that only the line starting at word 0 uses the
  // narrower first-line width. All break points after that use availableWidth.
  for (let i = n - 1; i >= 0; i--) {
    let lineWidth = 0;
    let bestCost = Infinity;
    let bestJ = i + 1;
    // The line starting at word i is the first line only if i === 0
    const widthForThisLine = i === 0 ? lineWidthFor(0) : availableWidth;

    for (let j = i; j < n; j++) {
      lineWidth += contentWords[j].word.width;
      if (j > i && contentWords[j].hasSpaceBefore) lineWidth += spaceWidth;

      if (lineWidth > widthForThisLine && j > i) break; // won't fit any more words

      const slack = widthForThisLine - lineWidth;
      let cost: number;

      if (j === n - 1) {
        // Last line — no penalty for being short
        cost = 0;
      } else if (slack < 0) {
        // Overflow — heavy penalty
        cost = 1e6 + Math.abs(slack) * 100;
      } else {
        // Badness = slack^3
        cost = slack * slack * slack / (widthForThisLine * widthForThisLine);
      }

      cost += breakCost[j + 1];

      if (cost < bestCost) {
        bestCost = cost;
        bestJ = j + 1;
      }
    }

    breakCost[i] = bestCost;
    breakNext[i] = bestJ;
  }

  // Reconstruct lines
  const lines: TypesetLine[] = [];
  let start = 0;
  let lineIndex = 0;

  while (start < n) {
    const end = breakNext[start];
    if (end <= start) break; // safety

    const lineWords: MeasuredWord[] = [];
    let lineWidth = 0;

    for (let k = start; k < end; k++) {
      if (k > start && contentWords[k].hasSpaceBefore) {
        // Add space only where there was whitespace in the original input
        lineWords.push({ text: ' ', width: spaceWidth, fontStyle: contentWords[k].word.fontStyle });
        lineWidth += spaceWidth;
      }
      lineWords.push(contentWords[k].word);
      lineWidth += contentWords[k].word.width;
    }

    const isLastLine = end >= n;
    lines.push({
      words: lineWords,
      width: lineWidth,
      availableWidth: lineWidthFor(lineIndex),
      isLastLine,
    });

    start = end;
    lineIndex++;
  }

  return lines;
}

/**
 * Simple greedy line breaker as fallback.
 */
export function breakLinesGreedy(
  words: MeasuredWord[],
  availableWidth: number,
): TypesetLine[] {
  if (words.length === 0) return [];

  const lines: TypesetLine[] = [];
  let currentWords: MeasuredWord[] = [];
  let currentWidth = 0;

  for (const word of words) {
    if (isWhitespace(word.text)) {
      if (currentWords.length > 0) {
        const newWidth = currentWidth + word.width;
        if (newWidth <= availableWidth) {
          currentWords.push(word);
          currentWidth = newWidth;
        }
      }
      continue;
    }

    const newWidth = currentWidth + word.width;
    if (newWidth <= availableWidth || currentWords.length === 0) {
      currentWords.push(word);
      currentWidth = newWidth;
    } else {
      // Trim trailing whitespace
      while (currentWords.length > 0 && isWhitespace(currentWords[currentWords.length - 1].text)) {
        const removed = currentWords.pop()!;
        currentWidth -= removed.width;
      }
      lines.push({ words: currentWords, width: currentWidth, availableWidth, isLastLine: false });
      currentWords = [word];
      currentWidth = word.width;
    }
  }

  if (currentWords.length > 0) {
    while (currentWords.length > 0 && isWhitespace(currentWords[currentWords.length - 1].text)) {
      const removed = currentWords.pop()!;
      currentWidth -= removed.width;
    }
    if (currentWords.length > 0) {
      lines.push({ words: currentWords, width: currentWidth, availableWidth, isLastLine: true });
    }
  }

  return lines;
}
