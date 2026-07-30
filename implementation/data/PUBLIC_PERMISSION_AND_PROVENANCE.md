# Public Permission and Provenance Record

```yaml
authorization_grantor_role: upstream_code_owner
authorization_status: received_and_privately_retained
authorized_scope: course_integration_and_public_redistribution_of_adapted_interface
attestation_recorded_by: project_owner
evidence_storage: retained_privately_outside_repository
evidence_available_to_instructor_if_required: true
adapted_upstream_repository: https://github.com/prestzy/OpenCV-Car-Parking
audited_commit: 12271576be39a4ac0eb456526eca122685799e8c
```

## Public code boundary

The adapted material is limited to the Stage W dashboard presentation:

- `implementation/src/parking_occupancy/stage_w_web/templates/dashboard.html`
- `implementation/src/parking_occupancy/stage_w_web/static/style.css`

The local project changed the presentation to show the unified backend's
annotated stream, occupied/vacant/total counts, attributed FPS, mode and cache
state, optional temporal/tracker state, and recent events. The local Flask
routes, synchronized worker, privacy redaction, CLI, and D1/B1/E1b/F2 backend
integration are project code. The upstream repository is not vendored.

This public record deliberately contains no participant identity, account,
conversation, screenshot, contact detail, authorization quotation, or private
evidence location. It records the role and authorized scope without implying
that the project owner granted rights owned by someone else. The older local
`STAGE_W_PERMISSION_AND_PROVENANCE.md` remains a non-public historical record
and is excluded from the W.3 public source manifest.
