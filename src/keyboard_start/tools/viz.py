import base64
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from argparse import ArgumentParser
from gzip import GzipFile
from io import BytesIO
from matplotlib.patches import FancyBboxPatch
from matplotlib.collections import PatchCollection, LineCollection


def normalize(s):
    return s.lower().replace('ё', 'е')


def drop_punctuation(s):
    return s.replace('-', '').replace("'", "")


def read_swipe_events(path, limit = -1):
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == limit:
                return

            ps = line.rstrip('\n').split('\t')
            curve = ps[0]
            word = "" if len(ps) == 1 else ps[1]
            try:
                content = base64.b64decode(curve)
                curve = GzipFile(fileobj=BytesIO(content), mode='rb').read()
            except Exception:
                pass
            j = json.loads(curve)
            word = word or j.get('word')
            yield (j, word)


def _get_whk(event):
    grid = event  # old version
    if 'curve' in event and 'grid' in event['curve']:
        grid = event['curve']['grid']
    return grid['width'], grid['height'], grid['keys']


def plot_keyboard(event, word=''):
    width, height, keys = _get_whk(event)

    fig = plt.figure(figsize=(30, 15))
    ax = fig.add_subplot(111)

    ax.set_xlim([0, width])
    ax.set_ylim([0, height])

    fig.subplots_adjust(
        top=0.86,
        bottom=0.09,
        left=0.100,
        right=0.925,
        hspace=0.2,
        wspace=0.2
    )

    patches = []
    pressed_patches = []

    for key in keys:
        h = key.get('background', key.get('hitbox'))
        p = FancyBboxPatch(
            (h['x'], h['y']),
             h['w'], h['h'],
            boxstyle="round,pad=1",
            fc='powderblue',
            ec='black', zorder=1
        )

        if key.get('label') and key['label'] in word:
            pressed_patches.append(p)
        else:
            patches.append(p)
        ch = key.get('label', '@')
        ch = ch if (ch and (ch.isalnum() or ch.isascii())) else '@'
        ax.text(h['x'] + 10, h['y'] + h['h'] - 10, ch, fontsize=20)

    collection = PatchCollection(patches)
    collection.set_facecolor("powderblue")
    collection.set_edgecolor("black")
    ax.add_collection(collection)

    pressed_collection = PatchCollection(pressed_patches)
    pressed_collection.set_facecolor("bisque")
    pressed_collection.set_edgecolor("black")
    ax.add_collection(pressed_collection)
    return fig, ax


def plot_swipe(event, ref, plot_ideal_curve=False, plot_speed=False, draw_only_reference=False):
    def get_key_center(key):
        h = key.get('background', key.get('hitbox'))
        return (h['x'] + h['w'] / 2, h['y'] + h['h'] / 2)

    width, height, keys = _get_whk(event)
    curve = event['curve']
    if not ref:
        ref = event.get('word', ref)

    label2center = dict()
    for key in keys:
        if 'label' not in key:
            continue
        cx, cy = get_key_center(key)
        label2center[key['label']] = (cx, cy)

    if 'ё' not in label2center:
        label2center['ё'] = label2center.get('е')

    if 'ъ' not in label2center:
        label2center['ъ'] = label2center.get('ь')

    ref_lower = ref.lower()
    ideal_curve_x = np.array([label2center[c][0] for c in ref_lower if c in label2center])
    ideal_curve_y = np.array([label2center[c][1] for c in ref_lower if c in label2center])

    # Plot keyboard.
    #
    fig, ax = plot_keyboard(event, ref_lower)

    # Calculate velocity.
    #
    x = np.array(curve['x'])
    y = np.array(curve['y'])
    t = np.array(curve['t'])

    if not len(x):
        print(f"Empty event, skip it. Word={ref}", file=sys.stderr)
        return fig, ax

    # Main drawing part.
    #
    if plot_ideal_curve:
        ax.plot(ideal_curve_x, ideal_curve_y, linewidth=3, color='salmon')
    if plot_speed:
        dt = t[1:] - t[:-1] + 0.5
        v = np.linalg.norm(np.array([x, y]).T[1:] - np.array([x, y]).T[:-1], axis=1) / dt
        max_v = np.max(v)

        lwidths = 1 + (v / max_v) ** 0.5 * 10
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, linewidths=lwidths, color='steelblue', zorder=10)
        ax.add_collection(lc)
    else:
        ax.plot(x, y, linewidth=3, color='steelblue')

    # Show timings.
    #
    for i in range(len(x)):
        ax.text(x[i], y[i], i, fontsize=10, zorder=10, color='black')

    ax.plot(x[0], y[0], 'go', markersize=12)
    ax.plot(x[-1], y[-1], 'ro', markersize=12)

    ax.set_xlim([0, width])
    ax.set_ylim([height, 0])

    # Print suggestion information.
    #
    def sy(y):
        return y / 1200 * height

    if 'context' in event:
        context = event['context']
        show = event.get('show', -1)
        oracle = event.get('oracle', False)
        unk = event.get('unk', False)
        ax.text(0, sy(-185), f'Context: "{context}" Reference: "{ref}" Time: {t[-1]}ms Show: {show} '
                             f'Unk: {unk} Oracle: {oracle}', fontsize=20)

    if 'result' in event:
        result = event['result']

        def draw_alignment(alignment, word, color, pos):
            xa = alignment['x']
            ya = alignment['y']
            ax.scatter(xa, ya, color=color, s=150, zorder=20 - pos, alpha=0.75)
            word = normalize(drop_punctuation(word))

            for i in range(len(xa)):
                ax.text(xa[i], ya[i] - sy(10), word[i], fontsize=25, zorder=30 - pos, color=color)

        COLORS = ('green', 'purple', 'red', 'gray', 'magenta', 'black')
        for i, (word, values) in enumerate(result.items(), start=1):
            alignment = values.pop('alignment', None)
            color = 'brown' if draw_only_reference else COLORS[i - 1]

            text = f'{i}: {word} {values}'
            ax.text(0, sy(-180 + 25 * i), text, fontsize=16, color=color)

            if not draw_only_reference and alignment:
                draw_alignment(alignment, word, color, i)

            values['alignment'] = alignment
            if i == len(COLORS):
                break

        if draw_only_reference:
            for i, (word, values) in enumerate(result.items(), start=1):
                if normalize(word) != normalize(ref):
                    continue

                alignment = values.pop('alignment', None)
                text = f'{i}: {word} {values}'
                i = min(len(COLORS) + 1, i)
                color = 'magenta'

                ax.text(0, sy(-180 + 25 * i), text, fontsize=16, color=color)

                if alignment:
                    draw_alignment(alignment, word, color, i)
                    break

    return fig, ax


def main():
    p = ArgumentParser()
    p.add_argument('-p', '--path', help='path to swipe events', required=True)
    p.add_argument('-o', '--output', help='output dict', default='curves')
    p.add_argument('--overwrite', help='overwrite figures', action='store_true')
    p.add_argument('--speed', help='use speed as width of lines', action='store_true')
    p.add_argument('--ideal', help='draw ideal curve', action='store_true')
    p.add_argument('-r', '--reference', help='draw only reference words', action='store_true')
    p.add_argument('-l', '--limit', help='pictures to draw', type=int, default=-1)
    args = p.parse_args()

    events = read_swipe_events(args.path, args.limit)
    os.makedirs(args.output, exist_ok=True)

    for i, (event, ref) in enumerate(events, start=1):
        name = f'{i:03d}'
        if ref:
            name = f'{name}_{ref}'
        name += '.png'
        save_path = os.path.join(args.output, name)
        print(f'Processing "{save_path}"')

        if os.path.exists(save_path) and not args.overwrite:
            continue

        fig, ax = plot_swipe(
            event, ref=ref, plot_ideal_curve=args.ideal,
            plot_speed=args.speed, draw_only_reference=args.reference
        )
        plt.savefig(save_path)
        plt.close(fig)


if __name__ == '__main__':
    main()
