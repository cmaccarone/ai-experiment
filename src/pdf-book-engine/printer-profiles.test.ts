import { describe, it, expect } from 'vitest';
import { LULU_PROFILE, KDP_PROFILE, INGRAM_PROFILE, lookupGutter } from './printer-profiles.js';

describe('lookupGutter', () => {
  it('returns correct gutter for small Lulu book', () => {
    expect(lookupGutter(LULU_PROFILE, 20)).toBe(0.375);
  });

  it('returns correct gutter for medium Lulu book', () => {
    expect(lookupGutter(LULU_PROFILE, 100)).toBe(0.5);
  });

  it('returns correct gutter for large Lulu book', () => {
    expect(lookupGutter(LULU_PROFILE, 350)).toBe(0.625);
  });

  it('returns correct gutter for very large Lulu book', () => {
    expect(lookupGutter(LULU_PROFILE, 500)).toBe(0.75);
  });

  it('returns last tier gutter for extremely large Lulu book', () => {
    expect(lookupGutter(LULU_PROFILE, 10000)).toBe(0.875);
  });

  it('works with KDP profile', () => {
    expect(lookupGutter(KDP_PROFILE, 10)).toBe(0.375);
    expect(lookupGutter(KDP_PROFILE, 200)).toBe(0.5);
    expect(lookupGutter(KDP_PROFILE, 400)).toBe(0.625);
  });

  it('works with IngramSpark profile', () => {
    expect(lookupGutter(INGRAM_PROFILE, 50)).toBe(0.5);
    expect(lookupGutter(INGRAM_PROFILE, 200)).toBe(0.625);
    expect(lookupGutter(INGRAM_PROFILE, 400)).toBe(0.75);
  });
});

describe('Printer profiles', () => {
  it('Lulu profile has expected structure', () => {
    expect(LULU_PROFILE.name).toBe('Lulu');
    expect(LULU_PROFILE.gutterTable.length).toBeGreaterThan(0);
    expect(LULU_PROFILE.maxPageCount).toBe(800);
  });

  it('KDP profile has expected structure', () => {
    expect(KDP_PROFILE.name).toBe('KDP');
    expect(KDP_PROFILE.maxPageCount).toBe(828);
  });

  it('IngramSpark profile has expected structure', () => {
    expect(INGRAM_PROFILE.name).toBe('IngramSpark');
    expect(INGRAM_PROFILE.maxPageCount).toBe(1050);
  });
});
