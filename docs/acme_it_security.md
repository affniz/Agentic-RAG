# Acme Corp IT Security Policy

## Passwords
Passwords must be at least 14 characters long and must not reuse any of your last 10 passwords.
Use the company password manager (1Password) for all work accounts; sharing passwords over chat or email is prohibited.
Passwords for standard accounts do not expire on a schedule, but must be changed immediately if a compromise is suspected.

## Multi-factor authentication
MFA is mandatory for email, VPN, the HR portal, and all production systems.
Hardware security keys (YubiKey) are required for engineers with production access; IT ships two keys to every such engineer.
SMS codes are not an accepted second factor.

## Laptops and devices
Company laptops must use full-disk encryption and lock automatically after 5 minutes of inactivity.
Operating system updates must be installed within 7 days of release; critical security patches within 48 hours.
Personal devices may access email and chat only through the managed mobile app; they may never hold customer data.

## Reporting incidents
Report suspected phishing with the "Report phishing" button in the mail client.
Lost or stolen devices must be reported to security@acme.example within 1 hour of discovery, 24/7.
Do not try to investigate a suspected breach yourself; isolate the device from the network and wait for the security team.

## Data classification
Data is classified as Public, Internal, Confidential, or Restricted.
Restricted data (customer payment data, health data, credentials) may only be stored in approved vaulted systems and never in spreadsheets or personal drives.
Confidential documents shared outside the company require a signed NDA.
