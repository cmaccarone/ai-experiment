import type { PrinterProfile } from './types.js';

export const LULU_PROFILE: PrinterProfile = {
  name: 'Lulu',
  gutterTable: [
    { maxPages: 32, gutter: 0.375 },
    { maxPages: 150, gutter: 0.5 },
    { maxPages: 400, gutter: 0.625 },
    { maxPages: 600, gutter: 0.75 },
    { maxPages: Infinity, gutter: 0.875 },
  ],
  minMargins: { top: 0.5, bottom: 0.5, inside: 0.375, outside: 0.25 },
  allowedTrimSizes: [
    { width: 5, height: 8, label: '5×8' },
    { width: 5.25, height: 8, label: '5.25×8' },
    { width: 5.5, height: 8.5, label: '5.5×8.5' },
    { width: 6, height: 9, label: '6×9' },
    { width: 6.14, height: 9.21, label: '6.14×9.21 (Royal)' },
    { width: 6.69, height: 9.61, label: '6.69×9.61 (Crown Quarto)' },
    { width: 7.44, height: 9.69, label: '7.44×9.69' },
    { width: 8.5, height: 11, label: '8.5×11 (Letter)' },
  ],
  maxPageCount: 800,
};

export const KDP_PROFILE: PrinterProfile = {
  name: 'KDP',
  gutterTable: [
    { maxPages: 24, gutter: 0.375 },
    { maxPages: 150, gutter: 0.375 },
    { maxPages: 300, gutter: 0.5 },
    { maxPages: 500, gutter: 0.625 },
    { maxPages: 700, gutter: 0.75 },
    { maxPages: Infinity, gutter: 0.875 },
  ],
  minMargins: { top: 0.25, bottom: 0.25, inside: 0.375, outside: 0.25 },
  allowedTrimSizes: [
    { width: 5, height: 8, label: '5×8' },
    { width: 5.25, height: 8, label: '5.25×8' },
    { width: 5.5, height: 8.5, label: '5.5×8.5' },
    { width: 6, height: 9, label: '6×9' },
    { width: 6.14, height: 9.21, label: '6.14×9.21' },
    { width: 7, height: 10, label: '7×10' },
    { width: 8.5, height: 11, label: '8.5×11' },
  ],
  maxPageCount: 828,
};

export const INGRAM_PROFILE: PrinterProfile = {
  name: 'IngramSpark',
  gutterTable: [
    { maxPages: 100, gutter: 0.5 },
    { maxPages: 250, gutter: 0.625 },
    { maxPages: 500, gutter: 0.75 },
    { maxPages: Infinity, gutter: 0.875 },
  ],
  minMargins: { top: 0.5, bottom: 0.5, inside: 0.5, outside: 0.375 },
  allowedTrimSizes: [
    { width: 5, height: 8, label: '5×8' },
    { width: 5.5, height: 8.5, label: '5.5×8.5' },
    { width: 6, height: 9, label: '6×9' },
    { width: 6.14, height: 9.21, label: '6.14×9.21' },
    { width: 8.5, height: 11, label: '8.5×11' },
  ],
  bleedRequired: false,
  maxPageCount: 1050,
};

/**
 * Look up the required gutter width for a given page count using a printer profile.
 */
export function lookupGutter(profile: PrinterProfile, pageCount: number): number {
  for (const entry of profile.gutterTable) {
    if (pageCount <= entry.maxPages) {
      return entry.gutter;
    }
  }
  // Fallback to last entry
  return profile.gutterTable[profile.gutterTable.length - 1].gutter;
}
