# Source the helper scripts listed in helpers/scripts, in the order listed.
# Paths in the manifest are relative to helpers/; the project root is found
# by `here` (via the `.here` file), so this works from any working directory
# inside the project.
for (script in readLines(here("helpers", "scripts"))) {
  source(here("helpers", script))
}
