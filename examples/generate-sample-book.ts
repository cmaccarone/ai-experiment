/**
 * Example: Generate a sample book PDF using the pdf-book-engine.
 *
 * Run with: npx tsx examples/generate-sample-book.ts
 * Output:   examples/sample-book.pdf
 */
import { readFileSync, writeFileSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { generateBook } from '../src/pdf-book-engine/index.js';

const __dirname = dirname(fileURLToPath(import.meta.url));

// Load fonts
const fontsDir = resolve(__dirname, 'fonts');
const bodyFont = readFileSync(resolve(fontsDir, 'EBGaramond-Regular.ttf'));
const italicFont = readFileSync(resolve(fontsDir, 'EBGaramond-Italic.ttf'));
const boldFont = readFileSync(resolve(fontsDir, 'EBGaramond-Bold.ttf'));

// Define chapters
const chapters = [
  {
    title: 'The Storm',
    html: `
      <p>It was a dark and stormy night. The wind howled through the ancient oaks
      that lined the drive, bending their branches into grotesque shapes against
      the churning sky. Rain hammered the windows of the old manor house with a
      fury that seemed almost personal, as if the storm itself bore some grudge
      against the inhabitants within.</p>

      <p>Eleanor stood at the window of the upstairs library, watching the
      lightning illuminate the garden in brief, violent flashes. Each bolt
      revealed the hedgerows and stone paths in stark white relief before
      plunging them back into darkness. She pressed her fingers against the
      cold glass and <em>waited</em>.</p>

      <p>"You shouldn't stand so close to the window," said a voice behind her.
      It was Marcus, her brother, carrying a candelabra that cast dancing shadows
      across the book-lined walls. "Lightning has been known to—"</p>

      <p>"I know what lightning does," Eleanor said, not turning around. "I'm
      waiting for the carriage. Father said he would arrive before nightfall."</p>

      <p>Marcus set the candelabra on the reading table and sank into the
      leather armchair beside it. "Father says many things," he replied, picking
      up a volume of <em>Plutarch's Lives</em> that lay open on the arm. "Not
      all of them come to pass."</p>

      <h2>The Arrival</h2>

      <p>It was nearly midnight when the sound of hooves on gravel cut through
      the storm. Eleanor flew down the stairs, her candle guttering in the
      draft from the front hall. The great oak door shuddered as someone
      pounded on it from outside.</p>

      <p>She drew back the bolt and pulled. The door swung inward, admitting
      a gust of wind and rain — and with it, a figure she did not recognize.
      The man was tall and gaunt, wrapped in a traveling cloak so sodden it
      seemed to weigh him down. Water pooled at his feet on the flagstone
      floor.</p>

      <p>"I beg your pardon," the stranger said, removing his hat to reveal
      a pale, angular face and dark eyes that caught the candlelight. "My
      carriage has thrown a wheel on the road below. Might I impose upon your
      hospitality until the storm passes?"</p>
    `,
  },
  {
    title: 'The Stranger',
    html: `
      <p>The stranger gave his name as Dr. Ashworth, a physician traveling
      north to attend a patient in the county. Eleanor showed him to the
      drawing room, where Marcus had built up the fire to a respectable blaze.</p>

      <p>Over brandy and cold meat, Dr. Ashworth proved to be a man of
      <strong>considerable</strong> learning and <em>peculiar</em> interests.
      He spoke of his travels through Eastern Europe, of ancient libraries
      he had visited in Prague and Budapest, and of certain manuscripts he
      had examined there — texts of great age that dealt with subjects most
      modern physicians would dismiss as superstition.</p>

      <h2>The Conversation</h2>

      <p>"You must understand," he said, leaning forward in his chair, the
      firelight casting deep shadows beneath his cheekbones, "that the
      distinction between science and what we call the <em>supernatural</em>
      is far more porous than our age would like to admit."</p>

      <p>Marcus, who had been listening with growing interest, set down his
      glass. "You speak as a man who has seen evidence of this porosity."</p>

      <p>"I speak as a man who has dedicated his life to understanding it."
      Dr. Ashworth reached into his coat and produced a small leather-bound
      notebook. "In here I have documented forty-seven cases — each one
      verified by multiple witnesses — of phenomena that no current theory
      of natural philosophy can adequately explain."</p>

      <h3>The Cases</h3>

      <p>He described several of these cases in detail:</p>

      <ul>
        <li>A woman in Bohemia who could predict earthquakes three days in advance</li>
        <li>A child in Transylvania who spoke fluent Latin without any instruction</li>
        <li>An elderly monk whose wounds healed at an impossible rate</li>
        <li>A village where every clock stopped simultaneously on the night of the equinox</li>
      </ul>

      <p>"Each case," Dr. Ashworth continued, "points to the same conclusion:
      there are forces at work in this world that operate according to laws
      we have not yet discovered. Not <em>supernatural</em> laws, mind you —
      simply <strong>natural laws</strong> that remain beyond our present
      understanding."</p>

      <p>Eleanor, who had been silent throughout this exchange, finally spoke.
      "And what has any of this to do with your journey north, Doctor?"</p>

      <p>Dr. Ashworth turned his dark eyes upon her, and for a moment she
      felt as if he were looking not at her face but at something behind it —
      something deep within. "Everything," he said quietly. "It has everything
      to do with it."</p>
    `,
  },
  {
    title: 'The Secret',
    html: `
      <p>The storm broke before dawn. Eleanor had not slept. She sat in the
      window seat of her bedroom, watching the clouds part to reveal a wash
      of pale stars, and thought about what Dr. Ashworth had told them after
      Marcus retired for the night.</p>

      <p>He had spoken of their father. Not by name — he was too careful for
      that — but the implication was unmistakable. The patient he traveled to
      see, the one whose condition he described with such grave concern, was
      none other than Sir Edmund Blackwood, their father, who had written
      to say he would be home before nightfall and had not come.</p>

      <p>"Your father's condition," Dr. Ashworth had said, choosing his words
      with the precision of a surgeon selecting instruments, "is unlike anything
      I have encountered in thirty years of medical practice. It does not
      respond to any treatment known to modern medicine. But I believe — and
      I stake my professional reputation on this — that it <em>will</em>
      respond to older methods. Methods that have been forgotten, or rather,
      <em>suppressed</em>, by those who fear what they do not understand."</p>

      <h2>The Decision</h2>

      <p>Eleanor watched the last stars fade as the sky lightened in the east.
      She had a choice to make, and she knew that everything — her father's
      life, the future of the estate, perhaps even her own understanding of
      what was possible in this world — depended upon it.</p>

      <p>She could dismiss Dr. Ashworth as a charlatan, a madman, or worse.
      Marcus would certainly counsel this course. He was a man of science, of
      the Royal Society, of measurable and repeatable results. He would say
      that their father needed proper medical attention, not the ministrations
      of a man who believed in forces beyond natural philosophy.</p>

      <p>Or she could trust the strange certainty she had felt when
      Dr. Ashworth looked at her with those dark, knowing eyes. The certainty
      that he was telling the truth — not the whole truth, perhaps, but
      enough of it to act upon.</p>

      <p>She rose from the window seat, smoothed her dress, and went
      downstairs to find the doctor. Her mind was made up.</p>

      <p>"I will go with you," she said, finding him already dressed and
      waiting in the front hall, his cloak dry, his hat in his hand, as if
      he had known exactly what she would decide and exactly when she would
      come to tell him. "Take me to my father."</p>

      <p>Dr. Ashworth inclined his head. "The carriage is repaired," he said.
      "We leave within the hour."</p>

      <p>And so they did, riding north through a landscape washed clean by
      the storm, toward whatever awaited them at the end of the road.</p>
    `,
  },
];

// Generate the PDF
console.log('Generating PDF...');
const startTime = Date.now();

const pdf = await generateBook(
  chapters,
  {
    trimWidth: 6,
    trimHeight: 9,
    margins: { top: 0.75, bottom: 0.75, inside: 0.75, outside: 0.5 },
    fonts: {
      body: bodyFont,
      bodyItalic: italicFont,
      bodyBold: boldFont,
      heading: boldFont,
    },
    fontSize: 11,
    lineHeight: 1.4,
    paragraphIndent: 1.5,
    chapterStartRecto: true,
    chapterTopDrop: 2,
    widowLines: 2,
    orphanLines: 2,
    header: (ctx) => {
      if (ctx.isChapterOpener) return null;
      if (ctx.pageNumber <= 1) return null;
      return {
        left: ctx.isRecto ? undefined : 'The Manor House',
        right: ctx.isRecto ? ctx.chapterTitle : undefined,
        font: 'bodyItalic',
        fontSize: 9,
      };
    },
    footer: (ctx) => ({
      center: String(ctx.pageNumber),
      fontSize: 10,
    }),
  },
  {},
);

const elapsed = Date.now() - startTime;
const outputPath = resolve(__dirname, 'sample-book.pdf');
writeFileSync(outputPath, pdf);
console.log(`Done in ${elapsed}ms — wrote ${pdf.length} bytes to ${outputPath}`);
