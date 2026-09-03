# Publish the DIII-D machine description

This repository has no CI workflow for machine-description publication. Publication is a local operator action. These commands are documentation only; the build command does not contact a registry.

Set the registry account and a token with package write permission:

```sh
GHCR_ACCOUNT='<registry-account>'
GHCR_TOKEN='<registry-token>'
printf '%s' "$GHCR_TOKEN" | oras login ghcr.io --username "$GHCR_ACCOUNT" --password-stdin
```

Push the canonical machine manifest as the OCI config and the netCDF IDS set as its machine-specific layer:

```sh
REFERENCE='ghcr.io/${GHCR_ACCOUNT}/diii-d-machine-description:dd-4.1.1-physical-a0a4959e38797180667ba802083e7ac833c9b2654441ecef1cb0dc1063d36e85'
ARTIFACT_TYPE='application/vnd.iter.nova.diii-d-machine-description.v1'
FILE_MEDIA_TYPE='application/vnd.iter.nova.diii-d-machine-description.ids.v1'
MANIFEST_PATH='docs/figures/diiid-forward-onboarding/ids-set/diiid_machine_description.manifest.json'
PAYLOAD_PATH='docs/figures/diiid-forward-onboarding/ids-set/diiid_machine_description.nc'
oras push --image-spec v1.1 --artifact-type "$ARTIFACT_TYPE" --config "$MANIFEST_PATH:$ARTIFACT_TYPE" "$REFERENCE" "$PAYLOAD_PATH:$FILE_MEDIA_TYPE"
```

Pull the layer and verify the exact payload bytes used to build the manifest:

```sh
REFERENCE='ghcr.io/${GHCR_ACCOUNT}/diii-d-machine-description:dd-4.1.1-physical-a0a4959e38797180667ba802083e7ac833c9b2654441ecef1cb0dc1063d36e85'
PULL_DIRECTORY='<pull-directory>'
mkdir -p "$PULL_DIRECTORY"
oras pull "$REFERENCE" --output "$PULL_DIRECTORY"
printf '%s  %s\n' 'a0a4959e38797180667ba802083e7ac833c9b2654441ecef1cb0dc1063d36e85' "$PULL_DIRECTORY/docs/figures/diiid-forward-onboarding/ids-set/diiid_machine_description.nc" | sha256sum --check -
```

The computed tag is `dd-4.1.1-physical-a0a4959e38797180667ba802083e7ac833c9b2654441ecef1cb0dc1063d36e85`. The OCI manifest media type is `application/vnd.oci.image.manifest.v1+json`; ORAS selects it through `--image-spec v1.1`.
