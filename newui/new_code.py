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

'''
def draw_phrase(canvas, parsed, x_start, y_start, row_height, font, font_scale, thickness, recording=False):
    # Bottom row: gloss words
    x = x_start
    word_positions = []
    for item in parsed:
        text = item['word']
        (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.putText(canvas, text, (x, y_start + 3 * row_height), font, font_scale, (0, 0, 0), thickness)
        word_positions.append((x, w))
        x += w + 25  # space between words

    # Define colors for each tag category
    tag_colors = [
        (0, 200, 0),   # grammar: green
        (0, 0, 255),   # mouth: red
        (150, 0, 150)  # emotion: purple
    ]

    # Draw annotation rows
    for row_idx, (category_tags, y_offset) in enumerate(zip(
            [grammar_tags, mouth_tags, emotion_tags],
            [y_start, y_start + row_height, y_start + 2 * row_height])):

        color = tag_colors[row_idx]
        spans = find_spans(parsed, category_tags)
        for span in spans:
            tag = span['tag']
            start_idx = span['start']
            end_idx = span['end']
            x_span_start = word_positions[start_idx][0]
            x_span_end = word_positions[end_idx][0] + word_positions[end_idx][1]

            # Draw tag text
            cv2.putText(canvas, tag, (x_span_start, y_offset), font, font_scale, color, thickness)

            # Draw underline spanning the rest of the span
            text_size = cv2.getTextSize(tag, font, font_scale, thickness)[0][0]
            line_start = (x_span_start + text_size + 5, y_offset + 5)
            line_end = (x_span_end, y_offset + 5)
            if line_end[0] > line_start[0]:
                cv2.line(canvas, line_start, line_end, color, thickness=2)

    # Optional: show recording status in top-right
    if not recording:
        cv2.putText(canvas, "(NOT RECORDING)", (canvas.shape[1] - 250, 40), font, 1, (0, 0, 255), 2)
'''
'''
import cv2

def draw_phrase(canvas, parsed, x_start, y_start, row_height, font, font_scale, thickness, recording=False):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2

    tag_definitions = [
        (grammar_tags, (0, 100, 0)),    # grammar: dark green
        (mouth_tags, (0, 0, 255)),      # mouth: red
        (emotion_tags, (150, 0, 150))   # emotion: purple
    ]

    num_rows = len(tag_definitions) + 1  # +1 for gloss row

    # Step 1: Calculate max width per word index (column)
    max_widths = []
    word_widths = []  # NEW: store actual word widths

    for i, item in enumerate(parsed):
        word = item['word']
        widths = []

        # Word width
        (word_w, _), _ = cv2.getTextSize(word, font, font_scale, thickness)
        word_widths.append(word_w)  # save actual word width
        widths.append(word_w)

        # Width of each annotation tag if present
        for tags, color in tag_definitions:
            matching_tag = next((tag for tag in tags if tag in item.get('annotations', [])), None)
            if matching_tag:
                (tag_w, _), _ = cv2.getTextSize(matching_tag, font, font_scale, thickness)
                widths.append(tag_w)
        
        max_widths.append(max(widths))

    # Add spacing between columns (words)
    space_between = 5
    # Calculate x positions for each word (column) cumulatively
    x_positions = []
    current_x = x_start
    for w in max_widths:
        x_positions.append(current_x)
        current_x += w + space_between

    # Step 2: Draw annotation tags rows and words
    for row_idx, (tags, color) in enumerate(tag_definitions):
        y_offset = y_start + row_height * row_idx
        spans = find_spans(parsed, tags)
        for span in spans:
            tag = span['tag']
            start_idx = span['start']
            end_idx = span['end']

            x_start_span = x_positions[start_idx]
            x_end_span = x_positions[end_idx] + max_widths[end_idx]

            # Draw tag label at start of span
            cv2.putText(canvas, tag, (x_start_span, y_offset), font, font_scale, color, thickness)

            # Draw underline from end of tag to end of last word's column
            tag_text_w = cv2.getTextSize(tag, font, font_scale, thickness)[0][0]
            line_start = (x_start_span + tag_text_w + 5, y_offset + 5)
            line_end = (x_positions[end_idx] + word_widths[end_idx], y_offset + 5)
            # line_end = (x_positions[end_idx] + max_widths[end_idx], y_offset + 5)
            # line_start = (x_start_span + tag_text_w + 5, y_offset + 5)
            # line_end = (x_end_span, y_offset + 5)
            if line_end[0] > line_start[0]:
                cv2.line(canvas, line_start, line_end, color, thickness=2)

    # Draw words in the bottom row
    bottom_y = y_start + row_height * (num_rows - 1)
    for i, item in enumerate(parsed):
        word = item['word']
        x = x_positions[i]
        cv2.putText(canvas, word, (x, bottom_y), font, font_scale, (0, 0, 0), thickness)
'''

# def draw_phrase(canvas, parsed, x_start, y_start, row_height, font, font_scale, thickness, recording=False):
#     tag_definitions = [
#         (grammar_tags, (0, 100, 0)),    # grammar: dark green
#         (mouth_tags, (0, 0, 255)),      # mouth: red
#         (emotion_tags, (150, 0, 150))   # emotion: purple
#     ]

#     num_rows = len(tag_definitions) + 1  # +1 for gloss row

#     # Step 1: Find all spans across categories
#     all_spans = []
#     for tags, _ in tag_definitions:
#         all_spans.extend(find_spans(parsed, tags))

#     # Step 2: Map start_idx to max annotation tag width at that index
#     annotation_start_widths = {}
#     for span in all_spans:
#         tag = span['tag']
#         start_idx = span['start']
#         tag_w = cv2.getTextSize(tag, font, font_scale, thickness)[0][0]
#         annotation_start_widths[start_idx] = max(annotation_start_widths.get(start_idx, 0), tag_w)

#     # Step 3: Calculate max width per word (column),
#     # including annotation width only if this word starts a span
#     max_widths = []
#     for i, item in enumerate(parsed):
#         word = item['word']
#         (word_w, _), _ = cv2.getTextSize(word, font, font_scale, thickness)
#         max_w = word_w
#         if i in annotation_start_widths:
#             max_w = max(max_w, annotation_start_widths[i])
#         max_widths.append(max_w)

#     # Add spacing between columns (words)
#     space_between = 25
#     x_positions = []
#     current_x = x_start
#     for w in max_widths:
#         x_positions.append(current_x)
#         current_x += w + space_between

#     # Step 4: Draw annotation tags rows and underline spans
#     for row_idx, (tags, color) in enumerate(tag_definitions):
#         y_offset = y_start + row_height * row_idx
#         spans = find_spans(parsed, tags)
#         for span in spans:
#             tag = span['tag']
#             start_idx = span['start']
#             end_idx = span['end']

#             x_start_span = x_positions[start_idx]
#             x_end_span = x_positions[end_idx] + max_widths[end_idx]

#             # Draw tag label only at span start
#             cv2.putText(canvas, tag, (x_start_span, y_offset), font, font_scale, color, thickness)

#             # Draw underline from end of tag text to end of last word in span
#             tag_text_w = cv2.getTextSize(tag, font, font_scale, thickness)[0][0]
#             line_start = (x_start_span + tag_text_w + 5, y_offset + 5)
#             line_end = (x_end_span, y_offset + 5)
#             if line_end[0] > line_start[0]:
#                 cv2.line(canvas, line_start, line_end, color, thickness=2)

#     # Step 5: Draw words in the bottom gloss row
#     bottom_y = y_start + row_height * (num_rows - 1)
#     for i, item in enumerate(parsed):
#         word = item['word']
#         x = x_positions[i]
#         cv2.putText(canvas, word, (x, bottom_y), font, font_scale, (0, 0, 0), thickness)

def draw_phrase(canvas, parsed, x_start, y_start, row_height, font, font_scale, thickness, recording=False):
    print("Drawing phrase...")

    tag_definitions = [
        (grammar_tags, (0, 100, 0)),
        (mouth_tags, (0, 0, 255)),
        (emotion_tags, (150, 0, 150))
    ]

    num_rows = len(tag_definitions) + 1  # +1 for gloss row

    all_spans = []
    for tags, _ in tag_definitions:
        all_spans.extend(find_spans(parsed, tags))

    annotation_start_widths = {}
    for span in all_spans:
        tag = span['tag']
        start_idx = span['start']
        tag_w = cv2.getTextSize(tag, font, font_scale, thickness)[0][0]
        annotation_start_widths[start_idx] = max(annotation_start_widths.get(start_idx, 0), tag_w)

    max_widths = []
    for i, item in enumerate(parsed):
        word = item['word']
        (word_w, _), _ = cv2.getTextSize(word, font, font_scale, thickness)
        max_w = word_w
        if i in annotation_start_widths:
            max_w = max(max_w, annotation_start_widths[i])
        max_widths.append(max_w)

    x_positions = []
    current_x = x_start
    space_between = 25

    for i, w in enumerate(max_widths):
        x_positions.append(current_x)
        current_x += w + space_between

    # Draw annotation tags rows and underline spans
    for row_idx, (tags, color) in enumerate(tag_definitions):
        y_offset = y_start + row_height * row_idx
        spans = find_spans(parsed, tags)
        for span in spans:
            tag = span['tag']
            start_idx = span['start']
            end_idx = span['end']

            x_start_span = x_positions[start_idx]
            x_end_span = x_positions[end_idx] + max_widths[end_idx]

            cv2.putText(canvas, tag, (x_start_span, y_offset), font, font_scale, color, thickness)

            tag_text_w = cv2.getTextSize(tag, font, font_scale, thickness)[0][0]
            line_start = (x_start_span + tag_text_w + 5, y_offset + 5)
            line_end = (x_end_span, y_offset + 5)
            if line_end[0] > line_start[0]:
                cv2.line(canvas, line_start, line_end, color, thickness=2)

    bottom_y = y_start + row_height * (num_rows - 1)
    for i, item in enumerate(parsed):
        word = item['word']
        x = x_positions[i]
        cv2.putText(canvas, word, (x, bottom_y), font, font_scale, (0, 0, 0), thickness)

import numpy as np
def main():
    # phrases = [
    #     "SHE(raise) fs-Marry(raise) ARRIVE(raise,pah) HOME(raise) LATE(raise) Mother MAYBE ANGRY(angry) SHE(raise)",
    #     "HE/SHE(raise) fs-John(raise,puff) LATE FAMILY(raise) DONTCARE",
    #     "WHEN(furrow) HE/SHE(mm) fs-JAMES(raise) DRIVE(raise,happy), WHY(furrow,happy) HE LATE",
    #     "eat(happy) sell(happy) where(furrow)",
    #     "eat(angry) sell(angry) where(furrow)",
    #     "eat(surprised) sell(surprised) where(furrow)",
    #     "no(shake) IX-center fs-Vivian not(shake) buy chocolate why(raise) IX-center sick"
    # ]
    phrases = [
        "IX-center fs-Xenos arrive home when(furrow)",
        "IX-center fs-Xenos arrive(pah) home when(furrow)",
        "eat sell where(furrow)",
        "eat(happy) sell(happy) where(furrow)",
        "eat(angry) sell(angry) where(furrow)",
        "eat(surprised) sell(surprised) where(furrow)",
        "fs-LOCKERBOX mother buy how-much(furrow)",
        "fs-LOCKERBOX mother buy(cs) how-much(furrow)",
        "when(raise) father(raise) read(raise) book(raise) mother read what(furrow)",
        "when(raise) father(raise) read(raise) book(raise,cha) mother read(pah) what(furrow)",
        "IX-center fs-JOHN-STEWART sick how(furrow)",
        "IX-center fs-JOHN-STEWART sick(cha) how(furrow)",
        "IX-center fs-JOHN-STEWART sick(oo) how(furrow)",
        "IX-center fs-JOHN-STEWART sick(mm) how(furrow)",
        "IX-center fs-RICK-PERRY disgust(disgust) why(furrow)",
        "IX-center(puff) fs-RICK-PERRY(puff) disgust(disgust) why(furrow)",
        "future go fs-CALIFORNIA who(furrow)",
        "future go(cs) fs-CALIFORNIA who(furrow)",
        "mother go store fs-zara which(furrow)",
        "mother go(pah) store(happy) fs-zara(happy) which(furrow)",
        "I surprised(surprised) why(raise) IX-center fs-nadal read book not_yet",
        "I surprised(surprised) why(raise) IX-center fs-nadal read book(mm) not_yet",
        "future family eat what(raise) fs-GAZPACHO",
        "future family eat(puff) what(raise) fs-GAZPACHO(happy)",
        "computer fs-MAC price how-much(raise) 30",
        "computer(oo) fs-MAC(oo) price how-much(raise) 30(surprised)",
        "IX-center fs-SHARPOVA go where(raise) my church",
        "IX-center fs-SHARPOVA go where(raise) my(cs) church(cs)",
        "my family go church when(raise) tomorrow",
        "my family(puff) go church(cha) when(raise) tomorrow",
        "arrive not_yet(th) who(raise) IX-center fs-Felix",
        "arrive not_yet(th) who(raise) IX-center(sad) fs-Felix(sad)",
        "arrive not_yet(th) who(raise) IX-center(angry) fs-Felix(angry)",
        "no(shake) IX-center fs-Vivian not(shake) buy chocolate why(raise) IX-center sick",
        "no(shake) IX-center fs-Vivian(puff) not(shake) buy(disgust) chocolate(disgust) why(raise) IX-center sick(cha)",
        "no(raise) when(raise) snow(raise) tomorrow(raise) class not(shake) cancel",
        "no(raise) when(raise) snow(raise) tomorrow(raise) class(cha) not(shake) cancel(scared)",
        "I angry(angry) if(raise) IX-center(raise) fs-HOWARD(raise) not(shake) clean kitchen",
        "I angry(angry) if(raise) IX-center(raise) fs-HOWARD(raise) not(shake) clean kitchen(oo)",
        "snow(raise) tomorrow(raise) store cancel(raise)",
        "snow(raise,puff) tomorrow(raise) store(mm) cancel(raise)",
        "store(raise) cancel(raise) I sad(sad)",
        "store(raise,oo) cancel(raise) I sad(sad)",
        "IX-center fs-JANET arrive late why(raise) snow",
        "IX-center fs-JANET arrive(sad) late(sad) why(raise) snow(puff)",
        "I surprised(surprised) why(raise) fs-xavier read book not_yet(th)",
        "I surprised(surprised) why(raise) fs-xavier read book(oo) not_yet(th)",
        "I scared(scared) why(raise) IX-center fs-BETH drive home late",
        "I scared(scared) why(raise) IX-center fs-BETH drive(th) home late",
        "when(raise) father(raise) go(raise) store(raise) IX-center fs-CAROL go class(raise)",
        "when(raise) father(raise) go(raise) store(raise) IX-center(surprised) fs-CAROL(surprised) go(surprised) class(raise)",
        "my friend fs-Felicity drive home(raise)",
        "my(scared) friend(scared) fs-Felicity(scared) drive(th) home(raise)",
        "father(raise) clean(raise) kitchen(raise) fs-Angelique happy(happy)",
        "father(raise) clean(raise,cs) kitchen(raise,cha) fs-Angelique happy(happy)"
    ]
    canvas_width = 1200
    canvas_height = 200
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    row_height = 40
    x_start = 20
    y_start = 40
    print(phrases)

    for phrase in phrases:
        canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
        parsed = parse_phrase(phrase)
        print(parsed)

        draw_phrase(canvas, parsed, x_start, y_start, row_height, font, font_scale, thickness)

        cv2.imshow("Phrase Display", canvas)
        print(f"Showing phrase: {phrase}")
        key = cv2.waitKey(0)
        if key == 27:  # ESC key to exit early
            break

    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()


# def main():
#     # phrase = "SHE(raise) fs-Marry(raise) ARRIVE(raise,pah) HOME(raise) LATE(raise) Mother MAYBE ANGRY(angry) SHE(raise)"
#     phrase = "HE/SHE(raise) fs-John(raise,puff) LATE FAMILY(raise) DONTCARE"
#     # phrase = "WHEN(furrow) HE/SHE(mm) fs-JAMES(raise) DRIVE(raise,happy), WHY(furrow,happy) HE LATE"
#     # phrase = "eat(happy) sell(happy) where(furrow)"
#     # phrase = "eat(angry) sell(angry) where(furrow)"
#     # phrase = "eat(surprised) sell(surprised) where(furrow)"
#     # phrase = "no(shake) IX-center fs-Vivian not(shake) buy chocolate why(raise) IX-center sick"

#     canvas_width = 1200
#     canvas_height = 200
#     canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 0.8
#     thickness = 2
#     row_height = 40
#     x_start = 20
#     y_start = 40

#     parsed = parse_phrase(phrase)

#     draw_phrase(canvas, parsed, x_start, y_start, row_height, font, font_scale, thickness)

#     cv2.imshow("Phrase Display", canvas)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()
