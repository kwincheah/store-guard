using System.Windows;
using System.Windows.Input;

namespace StoreGuard.Desktop.Views;

public partial class OwnerPinDialog : Window
{
    private readonly Func<string, Task<bool>>? _verifyPinAsync;
    private bool _isVerifying;

    public string EnteredPin { get; private set; } = string.Empty;

    public OwnerPinDialog() : this(null, false)
    {
    }

    public OwnerPinDialog(Func<string, Task<bool>>? verifyPinAsync, bool showDefaultPinHint)
    {
        _verifyPinAsync = verifyPinAsync;
        InitializeComponent();
        DefaultPinHintText.Visibility = showDefaultPinHint ? Visibility.Visible : Visibility.Collapsed;
        Loaded += (_, _) => PinBox.Focus();
    }

    private async void Unlock_Click(object sender, RoutedEventArgs e)
    {
        if (_isVerifying) return;
        var pin = PinBox.Password ?? string.Empty;

        if (string.IsNullOrWhiteSpace(pin))
        {
            ShowError("Enter your Admin PIN.");
            PinBox.Focus();
            return;
        }

        if (_verifyPinAsync is null)
        {
            EnteredPin = pin;
            DialogResult = true;
            return;
        }

        _isVerifying = true;
        UnlockButton.IsEnabled = false;
        try
        {
            bool valid = await _verifyPinAsync(pin);
            if (valid)
            {
                EnteredPin = pin;
                DialogResult = true;
            }
            else
            {
                ShowError("Incorrect Admin PIN. Try again.");
                PinBox.Clear();
                PinBox.Focus();
            }
        }
        finally
        {
            _isVerifying = false;
            UnlockButton.IsEnabled = true;
        }
    }

    private void Cancel_Click(object sender, RoutedEventArgs e) => DialogResult = false;

    private void PinBox_PasswordChanged(object sender, RoutedEventArgs e)
    {
        if (ErrorBanner.Visibility == Visibility.Visible && PinBox.Password.Length > 0)
            ErrorBanner.Visibility = Visibility.Collapsed;
    }

    private void PinBox_KeyDown(object sender, KeyEventArgs e)
    {
        if (e.Key == Key.Enter)
        {
            e.Handled = true;
            Unlock_Click(sender, e);
        }
    }

    public void ShowError(string message)
    {
        ErrorText.Text = message;
        ErrorBanner.Visibility = Visibility.Visible;
    }
}
