# Sourced by UCloud jobs: `upload FOLDER` copies a verified folder once to an immutable ERDA
# folder of the same name, using the dedicated key and known host from the secrets mount.
# Set MT_UPLOAD=0 to skip uploads.
erda_key="$(mktemp)"
install -m 600 /work/mini-trainer-secrets/erda-upload-key "$erda_key"
erda=(sftp -F /dev/null -i "$erda_key" -o IdentitiesOnly=yes -o IdentityAgent=none -o BatchMode=yes
    -o StrictHostKeyChecking=yes -o UserKnownHostsFile=/work/mini-trainer-secrets/erda-known-hosts
    -o GlobalKnownHostsFile=/dev/null -P 2222 -b - asgersvenning@ecos.au.dk@io.erda.au.dk)

upload() {
    local folder="$1" remote
    remote="/publications/hierarchical_classification/$(basename "$1")"
    [[ "${MT_UPLOAD:-1}" == 1 ]] || { echo "Upload skipped: $folder"; return 0; }
    [[ ! -e "$folder.uploaded" ]] || return 0
    printf 'mkdir %s\nput -r %s/* %s/\n' "$remote" "$folder" "$remote" | "${erda[@]}"
    printf 'get %s/manifest.json %s.remote-manifest.json\n' "$remote" "$folder" | "${erda[@]}"
    cmp "$folder.remote-manifest.json" "$folder/manifest.json"
    touch "$folder.uploaded"
    echo "Uploaded $remote"
}
