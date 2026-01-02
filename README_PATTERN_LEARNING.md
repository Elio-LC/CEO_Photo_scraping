# CEO Photo Pattern Learning

End-to-end workflow for learning CEO photo extraction rules from a small set of labeled samples.

## Goal
Label ~20 diverse DEF 14A filings, learn where and how CEO portraits appear, and generate optimized extraction rules that can be fed back into the pipeline.

## Step 1: Prepare labeling data
1. Create a seed list of filings to label. You can start from `samples_to_label.json` (generate your own stub if needed).
2. Populate `samples_to_label.json` with about 20 samples:
   ```json
   [
     {
       "company": "Apple Inc.",
       "cik": "0000320193",
       "fyear": 2023,
       "html_file": "D:/sec_data/apple_inc_0000320193/DEF_14A/xxx/primary-document.html"
     }
     // ...add ~20 rows total
   ]
   ```
3. Choose varied seeds:
   - Mix industries, fiscal years, and company sizes.
   - Confirm each HTML file exists and contains the CEO photo.

## Step 2: Interactive labeling
1. Run the labeler:
   ```bash
   python scrapephoto/label_ceo_photos.py samples_to_label.json ceo_photo_training_data.json
   ```
2. What happens:
   - The tool shows every image in each HTML.
   - You enter the index of the CEO photo (starting at 1).
   - Optional: add a note; enter `s` to skip the current sample.
   - `Ctrl+C` exits safely; progress is saved.
3. Output: `ceo_photo_training_data.json` with labeled training data.

## Step 3: Train the pattern learner
1. Run:
   ```bash
   python scrapephoto/ceo_photo_pattern_learner.py ceo_photo_training_data.json
   ```
2. Outputs in `pattern_learning_results/`:
   - `extracted_features.json` - all features extracted from labeled pages.
   - `learned_patterns.json` - consolidated patterns across samples.
   - `extraction_rules.py` - generated extraction heuristics.

## Step 4: Review the learning results
1. Console summary highlights (examples you may see):
   - CEO photos usually appear as image #2-8; average position ~4.3.
   - Typical size: width 200-400 px, height 250-500 px.
   - Frequent containers: `div` (~15), `td` (~8), `section` (~3).
   - Distance to keyword "SHAREHOLDERS": 500-3000 characters; median ~1200; present in 18/20 samples.
2. Deep dive: open `learned_patterns.json` to inspect:
   - Position distribution
   - Size ranges
   - Container patterns (common class/id)
   - Keyword distance stats
   - DOM path patterns

## Step 5: Apply the learned rules
- **Manual integration:** use `pattern_learning_results/extraction_rules.py` to refine `pipeline/ceo_photo_pipeline_test.py`:
  - Tighten image position filters.
  - Adjust size thresholds.
  - Add container class/id matches.
  - Tune keyword distance checks.
- **Automated (advanced):** read `learned_patterns.json` at runtime to adjust extraction parameters dynamically.

## Extracted features (per sample)
1. **Image position**
   - `image_position` (image index in the page)
   - `total_images`
2. **Image size**
   - `image_size` (width, height)
   - `has_width_height` (whether size attributes exist)
3. **Container hierarchy (per ancestor level)**
   - `parent_N_tag`, `parent_N_class`, `parent_N_id`
   - `parent_N_has_shareholders`, `parent_N_has_sincerely`
   - `parent_N_text_length`, `parent_N_img_count`
4. **Keyword distances**
   - `nearest_shareholders_distance`
   - `nearest_sincerely_distance`
   - `nearest_ceo_distance`
5. **DOM path**
   - `dom_path` (simplified path to the image)

## Example patterns often learned
1. **Location** - ~80% of CEO photos are images #2-6; rarely the first image.
2. **Size** - width typically 200-500 px; height 250-600 px; aspect ratio 0.6-1.2.
3. **Container** - parents are usually `div` (~70%), `td` (~20%), `section` (~10%); common classes: `signature`, `ceo-message`, `letter`; containers often include the words "shareholders" and "sincerely".
4. **Keyword distance** - distance to "sincerely" averages ~800 characters; distance to "to our shareholders" averages ~1500 characters; ~85% fall within 2000 characters of these keywords.
5. **DOM paths** - common patterns, e.g. `body > div.main > div.ceo-letter > img` or `body > table > tr > td > img`.

## Optimization tips for `find_ceo_letter_blocks()`
1. Add a position filter:
   ```python
   if img_position < 2 or img_position > 10:
       continue
   ```
2. Add a size filter:
   ```python
   if not (200 <= width <= 500 and 250 <= height <= 600):
       continue
   ```
3. Prioritize matches:
   - Higher priority if container class includes `signature`/`ceo`/`letter`.
   - Higher priority if within ~500 characters of keywords.
4. Strengthen container recognition: whitelist frequent class/id patterns surfaced by training.

## Continuous improvement
1. Expand training: label new samples regularly, retrain, and evolve the patterns.
2. A/B test: compare legacy rules vs. learned rules on held-out filings and track accuracy.
3. Outlier analysis: study extraction failures, add them to the training set, and refine thresholds.
