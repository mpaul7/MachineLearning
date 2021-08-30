

import sys
import chilkat

cert = chilkat.CkCert()

#  Load a DER (binary) encoded certificate.
#  To load from a .pem or .p7b, or any other file format that contains
#  just one certificate, call LoadFromFile in exactly the same way.
#  The LoadFromFile method automatically detects the format and loads the certificate.
success = cert.LoadFromFile("qa_data/certs/testCert.cer")
if (success == False):
    print(cert.lastErrorText())
    sys.exit()

#  Get the public key:
# pubKey is a CkPublicKey
pubKey = cert.ExportPublicKey()
if (cert.get_LastMethodSuccess() != True):
    print(cert.lastErrorText())
    sys.exit()

#  Examine the key type.
#  A PublicKey object can contain an RSA, ECC, or DSA public key.
#  The KeyType property will contain "rsa", "ecc", or "dsa".
print("key type = " + pubKey.keyType())