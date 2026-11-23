using System.Security.Cryptography;

namespace StoreGuard.Core.Utilities;

public static class PinHasher
{
    private const int SaltSize = 16;
    private const int KeySize = 32;
    private const int Iterations = 100_000;

    public static string HashPin(string pin)
    {
        var salt = RandomNumberGenerator.GetBytes(SaltSize);
        var hash = Rfc2898DeriveBytes.Pbkdf2(pin ?? string.Empty, salt, Iterations, HashAlgorithmName.SHA256, KeySize);
        return $"{Convert.ToBase64String(salt)}.{Convert.ToBase64String(hash)}";
    }

    public static bool VerifyPin(string pin, string storedHash)
    {
        if (string.IsNullOrWhiteSpace(storedHash))
            return false;
        var parts = storedHash.Split('.');
        if (parts.Length != 2)
            return false;
        try
        {
            var salt = Convert.FromBase64String(parts[0]);
            var expected = Convert.FromBase64String(parts[1]);
            var actual = Rfc2898DeriveBytes.Pbkdf2(pin ?? string.Empty, salt, Iterations, HashAlgorithmName.SHA256, KeySize);
            return CryptographicOperations.FixedTimeEquals(actual, expected);
        }
        catch
        {
            return false;
        }
    }
}
