//! Single-thread decode timing of an OMAC entry.
//!
//!     cargo run --release --example bench_decode -- file.omac [repeats]

use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let data = std::fs::read(&args[1]).expect("read file");
    let reps: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(5);
    let h = omac::Header::parse(&data).unwrap();
    let info = h.info;
    let c = info.channels as usize;
    let sel: Vec<usize> = (0..c).collect();
    let start = h.block_offsets[0] as usize;
    let mut out = vec![0f32; info.num_frames as usize * c];
    let mut best = f64::MAX;
    for _ in 0..reps {
        let t = Instant::now();
        omac::decode_range(&info, &data[start..], &h.block_offsets, 0, &sel, 0, info.num_frames, true, 1, &mut out)
            .unwrap();
        best = best.min(t.elapsed().as_secs_f64());
    }
    let audio_s = info.num_frames as f64 / info.sample_rate as f64;
    println!(
        "{} ch, {:.1} min: {:.1} ms, {:.0}x realtime, {:.0} Msamples/s",
        c,
        audio_s / 60.0,
        best * 1e3,
        audio_s / best,
        info.num_frames as f64 * c as f64 / best / 1e6
    );
}
