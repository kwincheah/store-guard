using System.Windows;
using System.Windows.Controls;

namespace StoreGuard.Desktop.Views;

public partial class SecretInputBox : UserControl
{
    public static readonly DependencyProperty SecretTextProperty =
        DependencyProperty.Register(
            nameof(SecretText),
            typeof(string),
            typeof(SecretInputBox),
            new FrameworkPropertyMetadata(string.Empty, FrameworkPropertyMetadataOptions.BindsTwoWayByDefault, OnSecretTextChanged));

    public static readonly DependencyProperty PlaceholderProperty =
        DependencyProperty.Register(nameof(Placeholder), typeof(string), typeof(SecretInputBox), new PropertyMetadata(string.Empty));

    public static readonly DependencyProperty IsRevealedProperty =
        DependencyProperty.Register(nameof(IsRevealed), typeof(bool), typeof(SecretInputBox), new PropertyMetadata(false, OnIsRevealedChanged));

    private bool _isSyncing;

    public SecretInputBox()
    {
        InitializeComponent();
        IsVisibleChanged += (_, e) =>
        {
            if (e.NewValue is false)
                IsRevealed = false;
        };
        Unloaded += (_, _) => IsRevealed = false;
        UpdateVisibilityAndPlaceholder();
    }

    public string SecretText
    {
        get => (string)(GetValue(SecretTextProperty) ?? string.Empty);
        set => SetValue(SecretTextProperty, value ?? string.Empty);
    }

    public string Placeholder
    {
        get => (string)(GetValue(PlaceholderProperty) ?? string.Empty);
        set => SetValue(PlaceholderProperty, value ?? string.Empty);
    }

    public bool IsRevealed
    {
        get => (bool)GetValue(IsRevealedProperty);
        set => SetValue(IsRevealedProperty, value);
    }

    private static void OnSecretTextChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
    {
        var ctl = (SecretInputBox)d;
        if (ctl._isSyncing) return;

        ctl._isSyncing = true;
        try
        {
            var next = e.NewValue as string ?? string.Empty;
            if (!string.Equals(ctl.SecretPasswordBox.Password, next, StringComparison.Ordinal))
                ctl.SecretPasswordBox.Password = next;
            if (!string.Equals(ctl.PlainTextBox.Text, next, StringComparison.Ordinal))
                ctl.PlainTextBox.Text = next;
            ctl.UpdateVisibilityAndPlaceholder();
        }
        finally
        {
            ctl._isSyncing = false;
        }
    }

    private static void OnIsRevealedChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
    {
        var ctl = (SecretInputBox)d;
        ctl.UpdateVisibilityAndPlaceholder();
    }

    private void SecretPasswordBox_PasswordChanged(object sender, RoutedEventArgs e)
    {
        if (_isSyncing) return;
        _isSyncing = true;
        try
        {
            var val = SecretPasswordBox.Password;
            PlainTextBox.Text = val;
            SecretText = val;
            UpdateVisibilityAndPlaceholder();
        }
        finally
        {
            _isSyncing = false;
        }
    }

    private void PlainTextBox_TextChanged(object sender, TextChangedEventArgs e)
    {
        if (_isSyncing) return;
        _isSyncing = true;
        try
        {
            var val = PlainTextBox.Text;
            SecretPasswordBox.Password = val;
            SecretText = val;
            UpdateVisibilityAndPlaceholder();
        }
        finally
        {
            _isSyncing = false;
        }
    }

    private void UpdateVisibilityAndPlaceholder()
    {
        bool revealed = IsRevealed;
        SecretPasswordBox.Visibility = revealed ? Visibility.Collapsed : Visibility.Visible;
        PlainTextBox.Visibility = revealed ? Visibility.Visible : Visibility.Collapsed;
        PlaceholderText.Visibility = string.IsNullOrEmpty(SecretText) ? Visibility.Visible : Visibility.Collapsed;
    }
}
