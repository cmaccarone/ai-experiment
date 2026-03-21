import type { TypesetLine } from './types.js';

/**
 * Page content item — anything that has a height and goes onto a page.
 */
export interface PageItem {
  type: 'line' | 'chapterTitle' | 'heading' | 'image' | 'blankSpace' | 'footnoteArea';
  height: number;
  // For line items
  typesetLine?: TypesetLine;
  // For chapter title / heading
  text?: string;
  level?: number;
  // For image items
  imageSrc?: string;
  imageWidth?: number;
  imageHeight?: number;
  // For footnote area
  footnoteLines?: TypesetLine[];
  // Can this item start a page?
  canStartPage?: boolean;
  // Keep with next item
  keepWithNext?: boolean;
}

export interface PageBreak {
  items: PageItem[];
  isChapterOpener: boolean;
  chapterTitle: string;
}

interface PageBreakConfig {
  availableHeight: number;  // in points
  widowLines: number;
  orphanLines: number;
}

/**
 * Assigns page items to pages, respecting widow/orphan constraints.
 *
 * A "widow" is a paragraph's last line appearing alone at the top of a page.
 * An "orphan" is a paragraph's first line appearing alone at the bottom of a page.
 */
export function breakPages(
  chapters: ChapterContent[],
  config: PageBreakConfig,
): PageBreak[] {
  const pages: PageBreak[] = [];

  for (const chapter of chapters) {
    // Start chapter on a new page
    if (pages.length > 0 || chapter.forceRecto) {
      // Add blank verso page if needed for recto start
      if (chapter.forceRecto && pages.length % 2 === 1) {
        pages.push({
          items: [],
          isChapterOpener: false,
          chapterTitle: chapter.title,
        });
      }
    }

    let currentItems: PageItem[] = [];
    let currentHeight = 0;
    let isFirstPageOfChapter = true;

    for (let i = 0; i < chapter.items.length; i++) {
      const item = chapter.items[i];

      // Check if adding this item would exceed page height
      if (currentHeight + item.height > config.availableHeight && currentItems.length > 0) {
        // Need a page break — but check widow/orphan constraints
        const adjusted = adjustForWidowOrphan(
          currentItems,
          chapter.items.slice(i),
          config,
          currentHeight,
        );

        pages.push({
          items: adjusted.pageitems,
          isChapterOpener: isFirstPageOfChapter,
          chapterTitle: chapter.title,
        });

        isFirstPageOfChapter = false;
        currentItems = adjusted.overflow;
        currentHeight = adjusted.overflow.reduce((h, it) => h + it.height, 0);

        // Re-add current item if it wasn't consumed
        if (!adjusted.consumed) {
          if (currentHeight + item.height <= config.availableHeight) {
            currentItems.push(item);
            currentHeight += item.height;
          } else {
            // Item alone exceeds page — force it onto a new page
            if (currentItems.length > 0) {
              pages.push({
                items: currentItems,
                isChapterOpener: false,
                chapterTitle: chapter.title,
              });
            }
            currentItems = [item];
            currentHeight = item.height;
          }
        }
      } else {
        currentItems.push(item);
        currentHeight += item.height;
      }
    }

    // Flush remaining items
    if (currentItems.length > 0) {
      pages.push({
        items: currentItems,
        isChapterOpener: isFirstPageOfChapter,
        chapterTitle: chapter.title,
      });
    }
  }

  return pages;
}

export interface ChapterContent {
  title: string;
  forceRecto: boolean;
  items: PageItem[];
}

interface AdjustResult {
  pageitems: PageItem[];
  overflow: PageItem[];
  consumed: boolean;  // whether the triggering item was already placed
}

function adjustForWidowOrphan(
  currentItems: PageItem[],
  remainingItems: PageItem[],
  config: PageBreakConfig,
  _currentHeight: number,
): AdjustResult {
  // Count trailing lines from the last paragraph on this page
  let trailingLines = 0;
  for (let i = currentItems.length - 1; i >= 0; i--) {
    if (currentItems[i].type === 'line') {
      trailingLines++;
    } else {
      break;
    }
  }

  // Orphan check: if the last paragraph only has a few lines at the bottom
  // and needs more (orphanLines), pull a line back
  if (trailingLines > 0 && trailingLines < config.orphanLines && currentItems.length > trailingLines) {
    // Move lines to make room — pull the orphaned line(s) to next page
    const overflow: PageItem[] = [];
    const linesToMove = config.orphanLines;
    const pageitems = currentItems.slice(0, currentItems.length - linesToMove);
    overflow.push(...currentItems.slice(currentItems.length - linesToMove));

    return { pageitems, overflow, consumed: false };
  }

  // Count leading lines of the next paragraph
  let leadingLines = 0;
  for (const item of remainingItems) {
    if (item.type === 'line') {
      leadingLines++;
    } else {
      break;
    }
  }

  // Widow check: if only 1-2 lines would start the next page, pull one back
  if (leadingLines > 0 && leadingLines < config.widowLines) {
    // Pull one line from this page to accompany the widow
    const overflow: PageItem[] = [];
    const pageitems = currentItems.slice(0, currentItems.length - 1);
    overflow.push(currentItems[currentItems.length - 1]);

    return { pageitems, overflow, consumed: false };
  }

  return { pageitems: currentItems, overflow: [], consumed: false };
}
