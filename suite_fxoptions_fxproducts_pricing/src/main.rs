// Main runner for all scripts

mod cody_algo;
mod amort_autorew_timeoption;
mod volretrieve_vanilla;
mod first_gen_exotic_px;
mod first_gen_exotic_px_2;
mod first_gen_exotic_px_3;
mod second_gen_exotic_px;
mod second_gen_exotic_px_2;
mod more_issues_fx_pricing_1;

fn main() {
    cody_algo::main();

    amort_autorew_timeoption::main();

    volretrieve_vanilla::main();

    first_gen_exotic_px::main();

    first_gen_exotic_px_2::main();

    first_gen_exotic_px_3::main();

    second_gen_exotic_px::main();

    second_gen_exotic_px_2::main();

    more_issues_fx_pricing_1::main();
}