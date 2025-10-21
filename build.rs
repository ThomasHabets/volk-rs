fn main() {
    // Ensures Cargo links against libvolk and picks up the right search paths.
    pkg_config::Config::new()
        .atleast_version("2.0") // adjust if needed
        .probe("volk")
        .expect("Could not find VOLK via pkg-config");
}

