import type { Block, StyledRun } from './types.js';

/**
 * Parses chapter HTML into a flat array of Blocks with styled runs.
 * Resolves all style inheritance during parsing so downstream stages never walk trees.
 *
 * Supported elements:
 *   Block: <p>, <h2>, <h3>, <ul>, <ol>, <li>, <img>
 *   Inline: <em>, <i>, <strong>, <b>, <br>, <a>
 */

interface StyleState {
  bold: boolean;
  italic: boolean;
  link?: string;
}

// Simple HTML tokenizer
interface Token {
  type: 'openTag' | 'closeTag' | 'selfCloseTag' | 'text';
  tag?: string;
  attrs?: Record<string, string>;
  text?: string;
}

function parseAttributes(attrString: string): Record<string, string> {
  const attrs: Record<string, string> = {};
  const re = /(\w[\w-]*)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|(\S+)))?/g;
  let match: RegExpExecArray | null;
  while ((match = re.exec(attrString)) !== null) {
    attrs[match[1]] = match[2] ?? match[3] ?? match[4] ?? '';
  }
  return attrs;
}

function tokenize(html: string): Token[] {
  const tokens: Token[] = [];
  const re = /<\s*\/\s*(\w+)\s*>|<\s*(\w+)((?:\s+[^>]*?)?)?\s*\/\s*>|<\s*(\w+)((?:\s+[^>]*?)?)\s*>|([^<]+)/g;
  let match: RegExpExecArray | null;

  while ((match = re.exec(html)) !== null) {
    if (match[1]) {
      // Close tag
      tokens.push({ type: 'closeTag', tag: match[1].toLowerCase() });
    } else if (match[2]) {
      // Self-closing tag
      tokens.push({
        type: 'selfCloseTag',
        tag: match[2].toLowerCase(),
        attrs: parseAttributes(match[3] ?? ''),
      });
    } else if (match[4]) {
      // Open tag
      tokens.push({
        type: 'openTag',
        tag: match[4].toLowerCase(),
        attrs: parseAttributes(match[5] ?? ''),
      });
    } else if (match[6]) {
      // Text node
      const text = decodeEntities(match[6]);
      if (text) {
        tokens.push({ type: 'text', text });
      }
    }
  }

  return tokens;
}

function decodeEntities(text: string): string {
  return sanitizeForWinAnsi(
    text
      .replace(/&amp;/g, '&')
      .replace(/&lt;/g, '<')
      .replace(/&gt;/g, '>')
      .replace(/&quot;/g, '"')
      .replace(/&#39;/g, "'")
      .replace(/&nbsp;/g, ' ')
      .replace(/&#(\d+);/g, (_, num) => String.fromCharCode(parseInt(num, 10)))
      .replace(/&#x([0-9a-fA-F]+);/g, (_, hex) => String.fromCharCode(parseInt(hex, 16))),
  );
}

/**
 * Replace Unicode characters that WinAnsi (CP1252) standard fonts cannot encode
 * with their closest ASCII/WinAnsi-safe equivalents.
 */
const UNICODE_REPLACEMENTS: Record<string, string> = {
  '\u2011': '-',   // non-breaking hyphen → hyphen-minus
  '\u2012': '-',   // figure dash → hyphen-minus
  '\u2013': '\u2013', // en dash — already in WinAnsi (0x96)
  '\u2014': '\u2014', // em dash — already in WinAnsi (0x97)
  '\u2015': '\u2014', // horizontal bar → em dash
  '\u2018': '\u2018', // left single quote — already in WinAnsi (0x91)
  '\u2019': '\u2019', // right single quote — already in WinAnsi (0x92)
  '\u201C': '\u201C', // left double quote — already in WinAnsi (0x93)
  '\u201D': '\u201D', // right double quote — already in WinAnsi (0x94)
  '\u2026': '\u2026', // ellipsis — already in WinAnsi (0x85)
  '\u2010': '-',   // hyphen → hyphen-minus
  '\u2043': '-',   // hyphen bullet → hyphen-minus
  '\u00AD': '-',   // soft hyphen → hyphen-minus (render as hyphen)
  '\u200B': '',    // zero-width space → remove
  '\u200C': '',    // zero-width non-joiner → remove
  '\u200D': '',    // zero-width joiner → remove
  '\uFEFF': '',    // BOM / zero-width no-break space → remove
  '\u2028': '\n',  // line separator → newline
  '\u2029': '\n',  // paragraph separator → newline
  '\u202F': ' ',   // narrow no-break space → space
  '\u205F': ' ',   // medium mathematical space → space
  '\u3000': ' ',   // ideographic space → space
};

function sanitizeForWinAnsi(text: string): string {
  // Fast path: check if any non-ASCII characters exist
  if (!/[^\x00-\x7F]/.test(text)) return text;

  let result = '';
  for (const char of text) {
    const replacement = UNICODE_REPLACEMENTS[char];
    if (replacement !== undefined) {
      result += replacement;
    } else {
      result += char;
    }
  }
  return result;
}

const BLOCK_TAGS = new Set(['p', 'h2', 'h3', 'ul', 'ol', 'li', 'img', 'br']);
const INLINE_TAGS = new Set(['em', 'i', 'strong', 'b', 'a', 'span']);

export function parseHtml(html: string): Block[] {
  const tokens = tokenize(html);
  const blocks: Block[] = [];
  let i = 0;

  function parseBlocks(): void {
    while (i < tokens.length) {
      const token = tokens[i];

      if (token.type === 'closeTag') {
        // Let parent handle
        return;
      }

      if (token.type === 'selfCloseTag') {
        if (token.tag === 'img') {
          blocks.push({ type: 'image', src: token.attrs?.src ?? '', alt: token.attrs?.alt });
          i++;
          continue;
        }
        if (token.tag === 'br') {
          blocks.push({ type: 'lineBreak' });
          i++;
          continue;
        }
        i++;
        continue;
      }

      if (token.type === 'openTag') {
        if (token.tag === 'p') {
          i++;
          const runs = collectInlineRuns({ bold: false, italic: false }, 'p');
          if (runs.length > 0) {
            blocks.push({ type: 'paragraph', runs });
          }
          skipCloseTag('p');
          continue;
        }

        if (token.tag === 'h2' || token.tag === 'h3') {
          const level = token.tag === 'h2' ? 2 : 3;
          i++;
          const runs = collectInlineRuns({ bold: false, italic: false }, token.tag);
          if (runs.length > 0) {
            blocks.push({ type: 'heading', level: level as 2 | 3, runs });
          }
          skipCloseTag(token.tag);
          continue;
        }

        if (token.tag === 'ul' || token.tag === 'ol') {
          const ordered = token.tag === 'ol';
          i++;
          const items = collectListItems();
          if (items.length > 0) {
            blocks.push({ type: 'list', ordered, items });
          }
          skipCloseTag(token.tag);
          continue;
        }

        if (token.tag === 'img') {
          blocks.push({ type: 'image', src: token.attrs?.src ?? '', alt: token.attrs?.alt });
          i++;
          // img might not have a close tag
          if (i < tokens.length && tokens[i].type === 'closeTag' && tokens[i].tag === 'img') {
            i++;
          }
          continue;
        }

        if (token.tag === 'br') {
          blocks.push({ type: 'lineBreak' });
          i++;
          if (i < tokens.length && tokens[i].type === 'closeTag' && tokens[i].tag === 'br') {
            i++;
          }
          continue;
        }

        // Unknown block tag — skip into it
        i++;
        parseBlocks();
        if (i < tokens.length && tokens[i].type === 'closeTag') {
          i++;
        }
        continue;
      }

      // Bare text outside block elements — wrap in paragraph
      if (token.type === 'text') {
        const trimmed = token.text?.trim();
        if (trimmed) {
          blocks.push({
            type: 'paragraph',
            runs: [{ text: token.text!, bold: false, italic: false }],
          });
        }
        i++;
        continue;
      }

      i++;
    }
  }

  function collectInlineRuns(style: StyleState, parentTag: string): StyledRun[] {
    const runs: StyledRun[] = [];

    while (i < tokens.length) {
      const token = tokens[i];

      if (token.type === 'closeTag') {
        if (token.tag === parentTag) {
          return runs;
        }
        // Mismatched close tag — return to parent
        return runs;
      }

      if (token.type === 'text') {
        const text = token.text!;
        if (text) {
          runs.push({
            text,
            bold: style.bold,
            italic: style.italic,
            ...(style.link ? { link: style.link } : {}),
          });
        }
        i++;
        continue;
      }

      if (token.type === 'selfCloseTag' && token.tag === 'br') {
        runs.push({ text: '\n', bold: false, italic: false });
        i++;
        continue;
      }

      if (token.type === 'openTag' && INLINE_TAGS.has(token.tag!)) {
        const tag = token.tag!;
        const newStyle = { ...style };

        if (tag === 'strong' || tag === 'b') newStyle.bold = true;
        if (tag === 'em' || tag === 'i') newStyle.italic = true;
        if (tag === 'a') newStyle.link = token.attrs?.href;

        i++;
        const innerRuns = collectInlineRuns(newStyle, tag);
        runs.push(...innerRuns);
        skipCloseTag(tag);
        continue;
      }

      // Unknown inline content — skip
      i++;
    }

    return runs;
  }

  function collectListItems(): StyledRun[][] {
    const items: StyledRun[][] = [];

    while (i < tokens.length) {
      const token = tokens[i];

      if (token.type === 'closeTag') {
        return items;
      }

      if (token.type === 'openTag' && token.tag === 'li') {
        i++;
        // ProseMirror/TipTap wraps li content in <p> tags — unwrap them
        if (i < tokens.length && tokens[i].type === 'openTag' && tokens[i].tag === 'p') {
          i++; // skip the inner <p>
        }
        const runs = collectInlineRuns({ bold: false, italic: false }, 'li');
        if (runs.length > 0) {
          items.push(runs);
        }
        // Skip any remaining close tags (</p> and/or </li>)
        while (i < tokens.length && tokens[i].type === 'closeTag' &&
               (tokens[i].tag === 'p' || tokens[i].tag === 'li')) {
          i++;
        }
        continue;
      }

      // Skip whitespace text between list items
      i++;
    }

    return items;
  }

  function skipCloseTag(tag: string): void {
    if (i < tokens.length && tokens[i].type === 'closeTag' && tokens[i].tag === tag) {
      i++;
    }
  }

  parseBlocks();
  return blocks;
}
