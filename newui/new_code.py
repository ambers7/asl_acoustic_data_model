import cv2
import numpy as np
import re

# Define categories
grammar_tags = {'raise', 'shake', 'furrow'}
mouth_tags = {'th', 'cs', 'cha', 'oo', 'mm', 'puff', 'ahh', 'pah'}
emotion_tags = {'happy', 'sad', 'angry', 'surprised', 'terrified', 'disgust'}

def parse_phrase(phrase_str):
    words = phrase_str.strip().split()
    parsed = []
    for w in words:
        m = re.match(r"([^\(]+)(?:\(([^)]+)\))?", w)
        word = m.group(1)
        annotations = m.group(2)
        anns = annotations.split(',') if annotations else []
        parsed.append({'word': word, 'annotations': [a.lower() for a in anns]})
    return parsed

def find_spans(parsed, category_tags):
    spans = []
    current_span = None
    for i, item in enumerate(parsed):
        anns = item['annotations']
        relevant = [a for a in anns if a in category_tags]
        if relevant:
            tag = relevant[0]
            if current_span is None or current_span['tag'] != tag:
                if current_span:
                    spans.append(current_span)
                current_span = {'tag': tag, 'start': i, 'end': i}
            else:
                current_span['end'] = i
        else:
            if current_span:
                spans.append(current_span)
                current_span = None
    if current_span:
        spans.append(current_span)
    return spans

def draw_phrase(canvas, parsed, x_start, y_start, row_height, font, font_scale, thickness):
    # Bottom row: gloss words
    x = x_start
    word_positions = []
    for item in parsed:
        text = item['word']
        (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.putText(canvas, text, (x, y_start + 3*row_height), font, font_scale, (0,0,225), thickness)
        word_positions.append((x, w))
        x += w + 25  # space between words

    # Draw annotation rows
    for row_idx, (category_tags, y_offset) in enumerate(zip(
            [grammar_tags, mouth_tags, emotion_tags],
            [y_start, y_start + row_height, y_start + 2*row_height])):
        
        spans = find_spans(parsed, category_tags)
        for span in spans:
            tag = span['tag']
            start_idx = span['start']
            end_idx = span['end']
            x_span_start = word_positions[start_idx][0]
            x_span_end = word_positions[end_idx][0] + word_positions[end_idx][1]

            # Draw tag text
            cv2.putText(canvas, tag, (x_span_start, y_offset), font, font_scale, (0,0,225), thickness)

            # Draw underline spanning rest of the span
            text_size = cv2.getTextSize(tag, font, font_scale, thickness)[0][0]
            line_start = (x_span_start + text_size + 5, y_offset + 5)
            line_end = (x_span_end, y_offset + 5)
            if line_end[0] > line_start[0]:
                cv2.line(canvas, line_start, line_end, (0,0,225), thickness=2)

def main():
    # phrase = "SHE(raise) fs-Marry(raise) ARRIVE(raise,pah) HOME(raise) LATE(raise) Mother MAYBE ANGRY(angry) SHE(raise)"
    # phrase = "HE/SHE(raise) fs-John(raise,puff) LATE FAMILY(raise) DONTCARE"
    phrase = "WHEN(furrow) HE/SHE(mm) fs-JAMES(raise) DRIVE(raise,happy), WHY(furrow,happy) HE LATE"


    canvas_width = 1200
    canvas_height = 200
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    row_height = 30
    x_start = 20
    y_start = 40

    parsed = parse_phrase(phrase)

    draw_phrase(canvas, parsed, x_start, y_start, row_height, font, font_scale, thickness)

    cv2.imshow("Phrase Display", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
