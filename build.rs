fn main() {
    if std::env::var("DOCS_RS").is_ok() {
        //println!("cargo:rustc-cfg=stub_volk");
        return;
    }
    // Ensures Cargo links against libvolk and picks up the right search paths.
    pkg_config::Config::new()
        .atleast_version("2.0") // adjust if needed
        .probe("volk")
        .expect("Could not find VOLK via pkg-config");
}
