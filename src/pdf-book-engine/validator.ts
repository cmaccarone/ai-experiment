import type { PdfBookConfig, PrinterProfile, ValidationWarning } from './types.js';
import { LULU_PROFILE } from './printer-profiles.js';

export function validateConfig(config: PdfBookConfig): ValidationWarning[] {
  const warnings: ValidationWarning[] = [];
  const printer: PrinterProfile = config.printer ?? LULU_PROFILE;

  // Validate trim size
  if (printer.allowedTrimSizes && printer.allowedTrimSizes.length > 0) {
    const match = printer.allowedTrimSizes.some(
      (ts) => ts.width === config.trimWidth && ts.height === config.trimHeight
    );
    if (!match) {
      const sizes = printer.allowedTrimSizes.map((ts) => ts.label ?? `${ts.width}×${ts.height}`).join(', ');
      warnings.push({
        code: 'UNSUPPORTED_TRIM_SIZE',
        message: `Trim size ${config.trimWidth}×${config.trimHeight} is not in ${printer.name}'s allowed sizes: ${sizes}`,
      });
    }
  }

  // Validate margins against minimums
  if (printer.minMargins) {
    const { top, bottom, inside, outside } = printer.minMargins;
    const m = config.margins;
    if (m.top < top)
      warnings.push({ code: 'MARGIN_TOO_SMALL', message: `Top margin ${m.top}" is below ${printer.name} minimum of ${top}"` });
    if (m.bottom < bottom)
      warnings.push({ code: 'MARGIN_TOO_SMALL', message: `Bottom margin ${m.bottom}" is below ${printer.name} minimum of ${bottom}"` });
    if (m.inside < inside)
      warnings.push({ code: 'MARGIN_TOO_SMALL', message: `Inside margin ${m.inside}" is below ${printer.name} minimum of ${inside}"` });
    if (m.outside < outside)
      warnings.push({ code: 'MARGIN_TOO_SMALL', message: `Outside margin ${m.outside}" is below ${printer.name} minimum of ${outside}"` });
  }

  return warnings;
}

export function validatePageCount(pageCount: number, printer: PrinterProfile): ValidationWarning[] {
  const warnings: ValidationWarning[] = [];
  if (printer.maxPageCount && pageCount > printer.maxPageCount) {
    warnings.push({
      code: 'PAGE_COUNT_EXCEEDED',
      message: `Total pages (${pageCount}) exceeds ${printer.name} maximum of ${printer.maxPageCount}`,
    });
  }
  return warnings;
}
