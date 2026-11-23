using System.Security.Cryptography;
using System.Text;
using StoreGuard.Core.Interfaces;

namespace StoreGuard.Infrastructure.Security;

public sealed class DpapiSecretProtector : ISecretProtector
{
    private const string Prefix = "dpapi:";

    public string Protect(string plaintext)
    {
        if (string.IsNullOrEmpty(plaintext))
            return string.Empty;
        if (plaintext.StartsWith(Prefix, StringComparison.Ordinal))
            return plaintext;

        var bytes = Encoding.UTF8.GetBytes(plaintext);
        var protectedBytes = ProtectedData.Protect(bytes, null, DataProtectionScope.CurrentUser);
        return Prefix + Convert.ToBase64String(protectedBytes);
    }

    public string Unprotect(string protectedText)
    {
        if (string.IsNullOrEmpty(protectedText))
            return string.Empty;
        if (!protectedText.StartsWith(Prefix, StringComparison.Ordinal))
            return protectedText;

        try
        {
            var cipher = Convert.FromBase64String(protectedText[Prefix.Length..]);
            var plainBytes = ProtectedData.Unprotect(cipher, null, DataProtectionScope.CurrentUser);
            return Encoding.UTF8.GetString(plainBytes);
        }
        catch
        {
            return string.Empty;
        }
    }
}
