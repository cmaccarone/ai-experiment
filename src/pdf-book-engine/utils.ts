// Unit conversions
export const POINTS_PER_INCH = 72;

export function inchesToPoints(inches: number): number {
  return inches * POINTS_PER_INCH;
}

export function pointsToInches(points: number): number {
  return points / POINTS_PER_INCH;
}

export function emsToPoints(ems: number, fontSize: number): number {
  return ems * fontSize;
}

// Page numbering: odd pages are recto (right-hand)
export function isRecto(pageNumber: number): boolean {
  return pageNumber % 2 === 1;
}
