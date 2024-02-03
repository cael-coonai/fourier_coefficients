use std::{ops::RangeInclusive, fs, f64::consts::PI};
use clap::Parser;
use rayon::prelude::*;

#[derive(Parser)]
#[command(name = "Fourier Series Coefficient Calculator")]
#[command(version = "0.1")]
#[command(about = ABOUT, long_about = None)]

struct Cli {
  #[arg(short = 'p', long, help = SHOW_PROGRESS_HELP)]
  show_progress: bool,
  #[arg(long, help = N0_HELP)]
  n0: Option<usize>,
  #[arg(help = N_HELP)]
  n: usize,
  #[arg(required = true, help = INPUT_FILES_HELP)]
  input_files: Vec<String>,
}

struct Args {
  show_progress: bool,
  n_range: RangeInclusive<usize>,
  n_min: usize,
  input_files: Vec<String>,
}

// const ABOUT:&str = "Calculates Fourier Series coefficients.";
const ABOUT: &str = "\
Calculates Fourier Series coefficients.

Fourier Coefficient Calculator calculates the coefficients for a Fourier Series
of a curve made up of discrete points from an input file. The calculator
generates a Fourier Series from the curve made by the linear interpolation of
those points. The calculator assumes that the points input are evenly spaced.
The coefficients are for the following form of the Fourier Series:

    f(t) = a_0/2 + Σ[a_n*cos(nt) + b_n*sin(nt)]\
";    
const SHOW_PROGRESS_HELP: &str ="Shows progress in calculation";
const N0_HELP: &str = "Lower limit for n (Default = 1)";
const N_HELP: &str = "Upper limit for n";
const INPUT_FILES_HELP: &str = "\
The expected format of the input is plain text file(s) with
values for f(t) on separate lines.\
";

fn get_args() -> Result<Args, String> {
  let cli = Cli::parse();

  if let Some(n0) = cli.n0 {
    if n0 > cli.n {
      return Err(format!("Value for n0 '{}' greater than n '{}'", n0, cli.n));
    }
  }

  let n_min = match cli.n0 {Some(x) => x, None => 1};
  let n_range = n_min..=cli.n;

  Ok(Args{
    show_progress: cli.show_progress,
    n_range,
    n_min,
    input_files: cli.input_files,
  })
}



fn parse_files(paths: Vec<String>, show_progress: bool) -> Vec<f64> {
  let mut data: Vec<f64> = vec![];
  for path in paths {
    if show_progress {println!("Reading file: {path}");}
    data.par_extend(
      fs::read_to_string(path.clone())
        .expect(format!("Failed to read file '{path}' with error").as_str())
        .par_lines()
        .map(|l| l.parse::<f64>()
          .expect(
            format!("Failed to read file '{path}' with error").as_str()
          )));
  }
  data.push(data[0]);
  data
}

fn compute_beta(data: &Vec<f64>, show_progress: bool) -> Vec<f64> {
  if show_progress {println!("Computing beta.")}
  data[0..data.len()-1]
    .par_iter()
    .zip(data[1..data.len()].par_iter())
    .map(|(a,b)| b-a)
    .collect()
}


fn compute_alpha(
  data: &Vec<f64>,
  betas: &Vec<f64>,
  show_progress: bool
) -> Vec<f64> {
  if show_progress {println!("Computing alpha.")}
  data[..data.len()-1]
    .par_iter()
    .zip(betas.par_iter())
    .zip((0..data.len()-1).into_par_iter())
    .map(|((val, beta), idx)| val - beta*(idx as f64))
    .collect()
}

fn compute_a_0(
  alphas: &Vec<f64>,
  betas: &Vec<f64>,
  show_progress: bool
) -> f64 {
  fn summand(alpha: &f64, beta: &f64, idx: usize) -> f64 {
    alpha * (idx as f64) + (0.5)*beta*(idx as f64).powi(2)
  }
  if show_progress {println!("Computing a_0.")}
  
  alphas.par_iter()
  .zip(betas.par_iter())
  .zip((0..alphas.len()).into_par_iter())
  .map(|((alpha, beta), idx)| {
    summand(alpha, beta, idx+1) - summand(alpha, beta, idx)
  })
  .sum::<f64>() * (2f64 / (alphas.len() as f64))
}

fn compute_a_n(
  alphas: &Vec<f64>,
  betas: &Vec<f64>,
  n_range: RangeInclusive<usize>,
  show_progress: bool
) -> Vec<f64> {
  fn summand(alpha: &f64, beta: &f64, n:usize, idx: usize, count: usize)-> f64 {
    let count = count as f64;
    let theta = (2f64 * PI * (n as f64) * (idx as f64)) / count;

    (count / (4f64*PI.powi(2)*(n as f64).powi(2))) * (
        2f64 * PI * (n as f64) * (alpha + beta * (idx as f64)) * theta.sin() +
        count * beta * theta.cos()
      )
  }
  if show_progress {println!("Computing a_n.")}
  
  n_range.into_par_iter()
    .map(|n| {
      alphas.par_iter()
        .zip(betas.par_iter())
        .zip((0..alphas.len()).into_par_iter())
        .map(|((alpha, beta), idx)| {
          summand(alpha, beta, n, idx+1, alphas.len()) -
          summand(alpha, beta, n, idx  , alphas.len())
        })
        .sum::<f64>() * (2f64 / (alphas.len() as f64))
    })
  .collect()  
}

fn compute_b_n(
  alphas: &Vec<f64>,
  betas: &Vec<f64>,
  n_range: RangeInclusive<usize>,
  show_progress: bool
) -> Vec<f64> {
  fn summand(alpha: &f64, beta: &f64, n:usize, idx: usize, count: usize)-> f64 {
    let count = count as f64;
    let theta = (2f64 * PI * (n as f64) * (idx as f64)) / count;

    (-1f64) * (count / (4f64*PI.powi(2)*(n as f64).powi(2))) * (
        2f64 * PI * (n as f64) * (alpha + beta * (idx as f64)) * theta.cos() -
        count * beta * theta.sin()
      )
  }
  if show_progress {println!("Computing b_n.")}
  
  n_range.into_par_iter()
    .map(|n| {
      alphas.par_iter()
        .zip(betas.par_iter())
        .zip((0..alphas.len()).into_par_iter())
        .map(|((alpha, beta), idx)| {
          summand(alpha, beta, n, idx+1, alphas.len()) -
          summand(alpha, beta, n, idx  , alphas.len())
        })
        .sum::<f64>() * (2f64 / (alphas.len() as f64))
    })
  .collect()
}


fn main() -> Result<(), String> {
  let args = get_args()?;

  let data = parse_files(args.input_files, args.show_progress);

  let beta = compute_beta(&data, args.show_progress);  
  let alpha = compute_alpha(&data, &beta, args.show_progress);

  let a_0= compute_a_0(&alpha, &beta, args.show_progress);
  let a_n= compute_a_n(&alpha, &beta, args.n_range.clone(), args.show_progress);
  let b_n= compute_b_n(&alpha, &beta, args.n_range.clone(), args.show_progress);

  println!("\na_0 =\n\"0{:E}\" ", a_0);
  println!("\na_n =");
  for n in args.n_range.clone() {
    print!("\"0{:E}\" ", a_n[n - args.n_min])
  }
  
  println!("\n\nb_n =");
  for n in args.n_range.clone() {
    print!("\"0{:E}\" ", b_n[n - args.n_min])
  }
  println!();
   
  return Ok(());  
}