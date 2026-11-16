# Incident Response Runbook

## Severity levels
- **SEV1**: complete outage of the Fleet API, or any confirmed customer data loss.
- **SEV2**: partial outage or significant degradation of the Fleet API or the dashboard affecting multiple customers.
- **SEV3**: minor issues with a workaround, or problems affecting a single customer.

## SEV1 response
Page the on-call incident commander (IC) within 5 minutes of detection.
Update the public status page within 15 minutes.
The Head of Customer Success owns customer communications for SEV1 incidents.
If the incident commander cannot be reached within 10 minutes, escalate to the VP Engineering.

## SEV2 and SEV3 response
For SEV2, notify the owning team's primary on-call within 30 minutes. SEV3 issues are handled during business hours through the team's ticket queue.

## After the incident
A written postmortem is due within 5 business days for SEV1 and SEV2 incidents.
