import { describe, it, expect } from 'vitest';
import { validateConfig, validatePageCount } from './validator.js';
import { LULU_PROFILE, KDP_PROFILE, INGRAM_PROFILE } from './printer-profiles.js';
import type { PdfBookConfig } from './types.js';

function makeConfig(overrides: Partial<PdfBookConfig> = {}): PdfBookConfig {
  return {
    trimWidth: 6,
    trimHeight: 9,
    margins: { top: 0.75, bottom: 0.75, inside: 0.75, outside: 0.5 },
    fonts: { body: 'dummy.ttf' },
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

describe('validateConfig', () => {
  it('returns no warnings for valid Lulu config', () => {
    const warnings = validateConfig(makeConfig({ printer: LULU_PROFILE }));
    expect(warnings).toHaveLength(0);
  });

  it('warns on unsupported trim size', () => {
    const warnings = validateConfig(makeConfig({
      printer: LULU_PROFILE,
      trimWidth: 4,
      trimHeight: 7,
    }));
    expect(warnings.some(w => w.code === 'UNSUPPORTED_TRIM_SIZE')).toBe(true);
  });

  it('warns when margins are below minimum', () => {
    const warnings = validateConfig(makeConfig({
      printer: LULU_PROFILE,
      margins: { top: 0.1, bottom: 0.1, inside: 0.1, outside: 0.1 },
    }));
    expect(warnings.filter(w => w.code === 'MARGIN_TOO_SMALL').length).toBeGreaterThanOrEqual(1);
  });

  it('validates against KDP profile', () => {
    const warnings = validateConfig(makeConfig({ printer: KDP_PROFILE }));
    expect(warnings).toHaveLength(0); // 6x9 is valid for KDP
  });

  it('validates against IngramSpark profile', () => {
    const warnings = validateConfig(makeConfig({ printer: INGRAM_PROFILE }));
    expect(warnings).toHaveLength(0);
  });
});

describe('validatePageCount', () => {
  it('returns no warnings when within limit', () => {
    const warnings = validatePageCount(400, LULU_PROFILE);
    expect(warnings).toHaveLength(0);
  });

  it('warns when page count exceeds maximum', () => {
    const warnings = validatePageCount(900, LULU_PROFILE);
    expect(warnings.some(w => w.code === 'PAGE_COUNT_EXCEEDED')).toBe(true);
  });
});
