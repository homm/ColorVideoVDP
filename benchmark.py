#!/usr/bin/env python

import argparse
import hashlib
import inspect
import io
import json
import math
import platform
import resource
import shutil
import statistics
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy
import PIL
from PIL import Image
import torch

import pycvvdp

if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
    import ffmpeg_binaries
    ffmpeg_binaries.add_to_path()


class Case(ABC):
    key: str
    meta: dict = {}
    registry = []

    def __init__(self, device):
        pass

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not inspect.isabstract(cls):
            Case.registry.append(cls)

    def setup(self):
        pass

    def warmup(self):
        self.run()

    @abstractmethod
    def run(self):
        pass


class CvvdpCase(Case):
    display = "standard_4k"

    def __init__(self, device):
        self.metric = pycvvdp.cvvdp(
            display_name=self.display, device=device, heatmap=None
        )
        self.meta = {"display": self.display}


class CvvdpImageCase(CvvdpCase):
    image_name: str

    def load_reference(self) -> Image.Image:
        source = Path(__file__).parent / "example_media" / self.image_name
        source_bytes = source.read_bytes()
        self.meta["source_sha256"] = hashlib.sha256(source_bytes).hexdigest()
        with Image.open(io.BytesIO(source_bytes)) as image:
            return image.convert("RGB")

    def make_jpeg(self, reference_image: Image.Image, quality) -> tuple[numpy.ndarray, str]:
        jpeg_buffer = io.BytesIO()
        reference_image.save(jpeg_buffer, format="JPEG", quality=quality)
        jpeg_hash = hashlib.sha256(jpeg_buffer.getbuffer()).hexdigest()
        jpeg_buffer.seek(0)
        with Image.open(jpeg_buffer) as image:
            return numpy.array(image.convert("RGB")), jpeg_hash

# -------------------------------- Cases -----------------------------------------------


class CvvdpConstructor(Case):
    key = "pycvvdp.cvvdp()"

    def __init__(self, device):
        self.device = device

    def run(self):
        pycvvdp.cvvdp(device=self.device, heatmap=None)
        return 1.0


class WavyFacadeJPEG75(CvvdpImageCase):
    image_name = "wavy_facade.png"
    key = f"CVVDP: {image_name} | JPEG q75"

    def setup(self):
        reference_image = self.load_reference()
        self.reference = numpy.array(reference_image)
        self.test, jpeg_hash = self.make_jpeg(reference_image, 75)

        self.meta.update({
            "jpeg_sha256": jpeg_hash,
            "jpeg_quality": 75,
        })

    def run(self):
        with torch.inference_mode():
            jod, _ = self.metric.predict(self.test, self.reference, dim_order="HWC")
        return float(jod)


class WavyFacadeJPEG75Cold(WavyFacadeJPEG75):
    key = f"{WavyFacadeJPEG75.key} cold"

    def warmup(self):
        pass

    def run(self):
        self.__init__(self.metric.device)
        return super().run()


Case.registry.remove(WavyFacadeJPEG75Cold)
Case.registry.insert(1, WavyFacadeJPEG75Cold)


class WavyFacadeJPEGBatch(CvvdpImageCase):
    image_name = "wavy_facade.png"
    qualities = range(30, 51, 5)
    key = f"CVVDP: {image_name} | JPEG q30-50 batch 5f"

    def setup(self):
        reference_image = self.load_reference()
        self.reference = numpy.array(reference_image)[None]

        tests, jpeg_hashes = zip(*(
            self.make_jpeg(reference_image, quality)
            for quality in self.qualities
        ))
        self.test = numpy.stack(tests)

        self.meta.update({
            "jpeg_sha256": list(jpeg_hashes),
            "jpeg_qualities": list(self.qualities),
        })

    def run(self):
        with torch.inference_mode():
            jods, _ = self.metric.predict(self.test, self.reference, dim_order="BHWC")
        return float(jods.mean())


class FerrisVideoSat(CvvdpCase):
    frames = 5
    key = f"CVVDP: ferris-ref.mp4 | ferris-test-sat.mp4 {frames}f"

    def setup(self):
        media = Path(__file__).parent / "example_media" / "structure"
        self.test = media / "ferris-test-sat.mp4"
        self.reference = media / "ferris-ref.mp4"
        self.metric.quiet = True
        self.meta.update({
            "test_sha256": hashlib.sha256(self.test.read_bytes()).hexdigest(),
            "reference_sha256": hashlib.sha256(self.reference.read_bytes()).hexdigest(),
        })

    def warmup(self):
        pass

    def run(self):
        source = pycvvdp.video_source_video_file(
            str(self.test), str(self.reference),
            display_photometry=self.metric.display_photometry, frames=self.frames,
        )
        with torch.inference_mode():
            result, _ = self.metric.predict_video_source(source)
        return float(result)


# -------------------------------- /Cases ----------------------------------------------


@dataclass
class Row:
    key: str
    current: dict
    previous: Optional[dict]
    delta: Optional[float]
    errors: list[str]


def environment(device):
    return {
        "device": str(device),
        "torch_threads": torch.get_num_threads(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "pillow_version": PIL.__version__,
    }


def measure(case, iterations):
    times = []
    values = []
    for _ in range(iterations):
        print(".", end="", file=sys.stderr, flush=True)
        started = time.perf_counter()
        result = case.run()
        times.append(time.perf_counter() - started)
        values.append(result)

    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{case.key}: the metric returned a non-finite result")
    return {
        **case.meta,
        "result": result,
        "times_seconds": times,
        "median_seconds": statistics.median(times),
    }, values


def load_results(path):
    if not path.exists():
        return None
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        print(f"Warning: discarding {path}: {error}", file=sys.stderr)
        return None
    if not isinstance(saved, dict) or not isinstance(saved.get("cases"), dict):
        print(f"Warning: discarding {path}: cases must be an object", file=sys.stderr)
        return None
    if saved["cases"] and "environment" not in saved:
        print(
            "Warning: saved environment is missing; it will be added on save",
            file=sys.stderr,
        )
    if "environment" in saved and (
        not isinstance(saved["environment"], dict)
        or not {"device", "torch_threads"} <= saved["environment"].keys()
    ):
        print(
            "Warning: saved environment is invalid; it will be replaced on save",
            file=sys.stderr,
        )
        saved.pop("environment")
    if "peak_rss_bytes" in saved and (
        type(saved["peak_rss_bytes"]) is not int
        or saved["peak_rss_bytes"] < 0
    ):
        print(
            "Warning: saved peak_rss_bytes is invalid; it will be replaced on save",
            file=sys.stderr,
        )
        saved.pop("peak_rss_bytes")
    return saved


def compare(previous, case, current):
    if (
        not isinstance(previous, dict)
        or not {*case.meta, "result", "median_seconds"} <= previous.keys()
    ):
        raise ValueError(f"{case.key}: saved case is missing required fields")
    for key in case.meta:
        if previous[key] != current[key]:
            raise ValueError(f"{case.key}: {key} differs from the saved result")
    old_result = previous["result"]
    old_time = previous["median_seconds"]
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in (old_result, old_time)
    ):
        raise ValueError(f"{case.key}: saved result or time is invalid")
    if not math.isfinite(old_result) or not math.isfinite(old_time) or old_time <= 0:
        raise ValueError(f"{case.key}: saved result or time is invalid")
    return current["result"] - old_result


def print_table(rows):
    headers = ("Case", "Result", "Old result", "Time, ms", "Old time, ms", "Errors")
    values = []
    for row in rows:
        old_result = (
            f'{row.previous["result"]:.3f} Δ{abs(row.delta):.4f}'
            if row.previous is not None else "—"
        )
        old_time = (
            f'{row.previous["median_seconds"] * 1000:.1f} '
            f'({row.previous["median_seconds"] / row.current["median_seconds"]:.2f}×)'
            if row.previous is not None else "—"
        )
        time_stdev = (
            statistics.stdev(row.current["times_seconds"])
            if len(row.current["times_seconds"]) > 1 else 0.0
        )
        current_time = (
            f'{row.current["median_seconds"] * 1000:.1f} '
            f'±{time_stdev * 1000:.1f}'
        )
        values.append((
            row.key, f'{row.current["result"]:.3f}', old_result,
            current_time, old_time, "; ".join(row.errors) or "—"
        ))

    widths = [
        max(len(row[column]) for row in (headers, *values))
        for column in range(len(headers))
    ]
    print("  ".join(cell.ljust(width) for cell, width in zip(headers, widths)))
    print("  ".join("─" * width for width in widths))
    for row in values:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)))


def save_results(path, results):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r", "--results", type=Path, default=Path("results.json"),
        help="Results JSON file (default: %(default)s)"
    )
    parser.add_argument(
        "-s", "--save", type=int, choices=(0, 1), default=None,
        help="Save results: 1=yes, 0=no (default: automatic or prompt)"
    )
    parser.add_argument(
        "-d", "--max-delta", type=float, default=0.01,
        help="Maximum absolute change from saved result (default: %(default)s)"
    )
    parser.add_argument(
        "-k", "--key",
        help="Run tests whose key contains this text (case-insensitive)"
    )
    parser.add_argument(
        "-n", "--iterations", type=int, default=3,
        help="Timed runs per test after warmup (default: %(default)s)"
    )
    parser.add_argument(
        "-c", "--concurrency", type=int, default=1,
        help="PyTorch CPU threads (default: %(default)s)"
    )
    args = parser.parse_args()
    if not math.isfinite(args.max_delta) or args.max_delta < 0:
        parser.error("--max-delta must be a non-negative finite number")
    if args.iterations < 1:
        parser.error("--iterations must be at least 1")
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")

    try:
        torch.set_num_threads(args.concurrency)
        case_classes = Case.registry
        if args.key is not None:
            key = args.key.casefold()
            case_classes = [
                case_class for case_class in case_classes
                if key in case_class.key.casefold()
            ]
        count = len(case_classes)
        if not case_classes:
            print(f"Running {count} tests", file=sys.stderr)
            return 1
        device = torch.device("cpu")
        saved = load_results(args.results)
        if saved is None:
            saved = {"cases": {}}
        current_environment = environment(device)
        if (saved_environment := saved.get("environment")) is not None:
            for key in ("device", "torch_threads"):
                previous = saved_environment[key]
                current = current_environment[key]
                if previous != current:
                    print(
                        f"Warning: {key} changed from {previous} to {current}",
                        file=sys.stderr,
                    )
        print(f"Running {count} test{'' if count == 1 else 's'}", file=sys.stderr)
        rows = []
        for case_class in case_classes:
            print(f"{case_class.key} ", end="", file=sys.stderr, flush=True)
            case = case_class(device)
            case.setup()
            case.warmup()
            current, values = measure(case, args.iterations)
            print(" DONE", file=sys.stderr)
            previous = saved["cases"].get(case.key)
            delta = None
            if case.key in saved["cases"]:
                try:
                    delta = compare(previous, case, current)
                except (ValueError, OverflowError) as error:
                    print(f"Warning: ignoring saved case: {error}", file=sys.stderr)
                    previous = None
            errors = []
            if any(value != values[0] for value in values[1:]):
                errors.append(
                    f"Result differs across runs ({min(values):.6f}–{max(values):.6f})"
                )
            if delta is not None and abs(delta) > args.max_delta:
                errors.append(f"Δ{delta:.4f} > ±{args.max_delta:.4f}")
            rows.append(Row(case.key, current, previous, delta, errors))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))

    print_table(rows)
    peak_rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        peak_rss_bytes *= 1024
    peak_rss = f"Peak RSS: {peak_rss_bytes / 1024**2:.1f} MiB"
    if (previous_peak := saved.get("peak_rss_bytes")) is not None:
        difference = (peak_rss_bytes - previous_peak) / 1024**2
        peak_rss += f" (Δ {difference:+.1f} MiB)"
    print(peak_rss, file=sys.stderr)
    if any(row.errors for row in rows):
        return 1

    if args.save is None:
        if not args.results.exists():
            save = True
        elif sys.stdin.isatty():
            choice = input(f"Save result to {args.results}? [y/N]")
            save = choice.strip().lower() in ("y", "yes")
        else:
            save = False
    else:
        save = bool(args.save)

    if save:
        saved["environment"] = current_environment
        saved["peak_rss_bytes"] = peak_rss_bytes
        for row in rows:
            saved["cases"][row.key] = row.current
        save_results(args.results, saved)
        print(f"Saved to {args.results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
