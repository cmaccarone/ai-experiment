import { describe, it, expect } from 'vitest';
import { inchesToPoints, pointsToInches, emsToPoints, isRecto, POINTS_PER_INCH } from './utils.js';

describe('unit conversions', () => {
  it('converts inches to points', () => {
    expect(inchesToPoints(1)).toBe(72);
    expect(inchesToPoints(0.5)).toBe(36);
    expect(inchesToPoints(6)).toBe(432);
  });

  it('converts points to inches', () => {
    expect(pointsToInches(72)).toBe(1);
    expect(pointsToInches(36)).toBe(0.5);
  });

  it('converts ems to points', () => {
    expect(emsToPoints(1.5, 11)).toBe(16.5);
    expect(emsToPoints(1, 12)).toBe(12);
  });

  it('POINTS_PER_INCH is 72', () => {
    expect(POINTS_PER_INCH).toBe(72);
  });
});

describe('isRecto', () => {
  it('odd pages are recto', () => {
    expect(isRecto(1)).toBe(true);
    expect(isRecto(3)).toBe(true);
    expect(isRecto(101)).toBe(true);
  });

  it('even pages are verso', () => {
    expect(isRecto(2)).toBe(false);
    expect(isRecto(4)).toBe(false);
    expect(isRecto(100)).toBe(false);
  });
});
