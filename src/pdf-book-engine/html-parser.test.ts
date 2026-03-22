import { describe, it, expect } from 'vitest';
import { parseHtml } from './html-parser.js';

describe('parseHtml', () => {
  it('parses a simple paragraph', () => {
    const blocks = parseHtml('<p>Hello world</p>');
    expect(blocks).toEqual([
      { type: 'paragraph', runs: [{ text: 'Hello world', bold: false, italic: false }] },
    ]);
  });

  it('parses nested inline styles (em + strong)', () => {
    const blocks = parseHtml('<p>She said <em>this is <strong>very</strong> important</em> to me.</p>');
    expect(blocks).toEqual([
      {
        type: 'paragraph',
        runs: [
          { text: 'She said ', bold: false, italic: false },
          { text: 'this is ', bold: false, italic: true },
          { text: 'very', bold: true, italic: true },
          { text: ' important', bold: false, italic: true },
          { text: ' to me.', bold: false, italic: false },
        ],
      },
    ]);
  });

  it('parses headings', () => {
    const blocks = parseHtml('<h2>Section Title</h2>');
    expect(blocks).toEqual([
      { type: 'heading', level: 2, runs: [{ text: 'Section Title', bold: false, italic: false }] },
    ]);
  });

  it('parses h3', () => {
    const blocks = parseHtml('<h3>Sub Section</h3>');
    expect(blocks).toEqual([
      { type: 'heading', level: 3, runs: [{ text: 'Sub Section', bold: false, italic: false }] },
    ]);
  });

  it('parses unordered list', () => {
    const blocks = parseHtml('<ul><li>First</li><li>Second</li></ul>');
    expect(blocks).toEqual([
      {
        type: 'list',
        ordered: false,
        items: [
          [{ text: 'First', bold: false, italic: false }],
          [{ text: 'Second', bold: false, italic: false }],
        ],
      },
    ]);
  });

  it('parses ordered list', () => {
    const blocks = parseHtml('<ol><li>One</li><li>Two</li></ol>');
    expect(blocks).toEqual([
      {
        type: 'list',
        ordered: true,
        items: [
          [{ text: 'One', bold: false, italic: false }],
          [{ text: 'Two', bold: false, italic: false }],
        ],
      },
    ]);
  });

  it('parses self-closing img', () => {
    const blocks = parseHtml('<img src="map.png" alt="A map" />');
    expect(blocks).toEqual([
      { type: 'image', src: 'map.png', alt: 'A map' },
    ]);
  });

  it('parses links with href', () => {
    const blocks = parseHtml('<p>Visit <a href="https://example.com">here</a> for more.</p>');
    expect(blocks).toHaveLength(1);
    expect(blocks[0]).toHaveProperty('type', 'paragraph');
    const runs = (blocks[0] as any).runs;
    expect(runs).toHaveLength(3);
    expect(runs[1]).toEqual({ text: 'here', bold: false, italic: false, link: 'https://example.com' });
  });

  it('handles multiple paragraphs', () => {
    const blocks = parseHtml('<p>First paragraph.</p><p>Second paragraph.</p>');
    expect(blocks).toHaveLength(2);
    expect(blocks[0]).toHaveProperty('type', 'paragraph');
    expect(blocks[1]).toHaveProperty('type', 'paragraph');
  });

  it('decodes HTML entities', () => {
    const blocks = parseHtml('<p>Tom &amp; Jerry &lt;3&gt;</p>');
    const runs = (blocks[0] as any).runs;
    expect(runs[0].text).toBe('Tom & Jerry <3>');
  });

  it('handles bold with <b> tag', () => {
    const blocks = parseHtml('<p><b>bold text</b></p>');
    const runs = (blocks[0] as any).runs;
    expect(runs[0]).toEqual({ text: 'bold text', bold: true, italic: false });
  });

  it('handles italic with <i> tag', () => {
    const blocks = parseHtml('<p><i>italic text</i></p>');
    const runs = (blocks[0] as any).runs;
    expect(runs[0]).toEqual({ text: 'italic text', bold: false, italic: true });
  });

  it('sanitizes non-breaking hyphen (U+2011) to regular hyphen', () => {
    const blocks = parseHtml('<p>self\u2011driving cars</p>');
    const runs = (blocks[0] as any).runs;
    expect(runs[0].text).toBe('self-driving cars');
  });

  it('sanitizes non-breaking hyphen from HTML entity &#8209;', () => {
    const blocks = parseHtml('<p>well&#8209;known</p>');
    const runs = (blocks[0] as any).runs;
    expect(runs[0].text).toBe('well-known');
  });

  it('removes zero-width characters', () => {
    const blocks = parseHtml('<p>hello\u200Bworld</p>');
    const runs = (blocks[0] as any).runs;
    expect(runs[0].text).toBe('helloworld');
  });

  it('replaces narrow no-break space with regular space', () => {
    const blocks = parseHtml('<p>100\u202Fkg</p>');
    const runs = (blocks[0] as any).runs;
    expect(runs[0].text).toBe('100 kg');
  });

  it('preserves WinAnsi-safe characters like en dash and em dash', () => {
    const blocks = parseHtml('<p>2020\u20132025 \u2014 a period</p>');
    const runs = (blocks[0] as any).runs;
    expect(runs[0].text).toBe('2020\u20132025 \u2014 a period');
  });
});
