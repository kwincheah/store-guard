using System.Windows;

namespace StoreGuard.Desktop.Views;

public partial class ConfirmDialog : Window
{
    public ConfirmDialog(string title, string message, string confirmText, bool isDestructive)
    {
        InitializeComponent();
        Title = $"{title} — StoreGuard";
        TitleText.Text = title;
        MessageText.Text = message;
        ConfirmButton.Content = string.IsNullOrWhiteSpace(confirmText) ? "Confirm" : confirmText;

        if (!isDestructive)
        {
            ConfirmButton.Style = (Style)FindResource("PrimaryButtonStyle");
            IconBadge.SetResourceReference(System.Windows.Controls.Border.BackgroundProperty, "PrimaryTintBrush");
            IconBadge.SetResourceReference(System.Windows.Controls.Border.BorderBrushProperty, "PrimaryBorderBrush");
            IconGlyph.SetResourceReference(System.Windows.Shapes.Shape.StrokeProperty, "PrimaryTextBrush");
        }
    }

    private void Confirm_Click(object sender, RoutedEventArgs e) => DialogResult = true;
    private void Cancel_Click(object sender, RoutedEventArgs e) => DialogResult = false;
}
