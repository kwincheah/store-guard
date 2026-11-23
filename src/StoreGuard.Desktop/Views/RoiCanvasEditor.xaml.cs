using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;
using StoreGuard.Core.Models;
using StoreGuard.Core.Utilities;
using StoreGuard.Desktop.ViewModels;

namespace StoreGuard.Desktop.Views;

public partial class RoiCanvasEditor : UserControl
{
    private enum DragMode { None, DrawNew, MoveZone, ResizeCorner }
    private enum CornerHandle { None, TopLeft, TopRight, BottomLeft, BottomRight }

    public static readonly DependencyProperty SnapshotBytesProperty =
        DependencyProperty.Register(nameof(SnapshotBytes), typeof(byte[]), typeof(RoiCanvasEditor),
            new PropertyMetadata(null, OnSnapshotBytesChanged));

    public static readonly DependencyProperty ZonesProperty =
        DependencyProperty.Register(nameof(Zones), typeof(ObservableCollection<EditableZoneViewModel>), typeof(RoiCanvasEditor),
            new PropertyMetadata(null, OnZonesChanged));

    public static readonly DependencyProperty SelectedZoneProperty =
        DependencyProperty.Register(nameof(SelectedZone), typeof(EditableZoneViewModel), typeof(RoiCanvasEditor),
            new FrameworkPropertyMetadata(null, FrameworkPropertyMetadataOptions.BindsTwoWayByDefault, OnSelectedZoneChanged));

    public static readonly DependencyProperty ActiveDrawRoleProperty =
        DependencyProperty.Register(nameof(ActiveDrawRole), typeof(string), typeof(RoiCanvasEditor),
            new PropertyMetadata("shelf"));

    public event Action<double[]>? ZoneDrawn;
    public event Action? BeforeZoneMutation;
    public event Action? ZoneModified;

    private DragMode _dragMode = DragMode.None;
    private CornerHandle _activeCorner = CornerHandle.None;
    private Point? _dragStart;
    private EditableZoneViewModel? _dragTargetZone;
    private double[]? _dragOrigRectPx;
    private bool _pushedUndoForCurrentDrag;
    private Rectangle? _previewRect;
    private Border? _previewTag;
    private double _imgW = 1280;
    private double _imgH = 720;

    public byte[]? SnapshotBytes
    {
        get => (byte[]?)GetValue(SnapshotBytesProperty);
        set => SetValue(SnapshotBytesProperty, value);
    }

    public ObservableCollection<EditableZoneViewModel>? Zones
    {
        get => (ObservableCollection<EditableZoneViewModel>?)GetValue(ZonesProperty);
        set => SetValue(ZonesProperty, value);
    }

    public EditableZoneViewModel? SelectedZone
    {
        get => (EditableZoneViewModel?)GetValue(SelectedZoneProperty);
        set => SetValue(SelectedZoneProperty, value);
    }

    public string ActiveDrawRole
    {
        get => (string)(GetValue(ActiveDrawRoleProperty) ?? "shelf");
        set => SetValue(ActiveDrawRoleProperty, value);
    }

    public RoiCanvasEditor()
    {
        InitializeComponent();
        UpdateEmptyHint();
    }

    private void UpdateEmptyHint()
    {
        EmptyHintOverlay.Visibility = (SnapshotImg.Source is not null || (Zones is not null && Zones.Count > 0))
            ? Visibility.Collapsed
            : Visibility.Visible;
    }

    private static void OnSnapshotBytesChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
        => ((RoiCanvasEditor)d).LoadSnapshot(e.NewValue as byte[]);

    private static void OnSelectedZoneChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
        => ((RoiCanvasEditor)d).RedrawZones();

    private static void OnZonesChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
    {
        var ed = (RoiCanvasEditor)d;
        if (e.OldValue is INotifyCollectionChanged oldCol)
            oldCol.CollectionChanged -= ed.Zones_CollectionChanged;
        if (e.NewValue is INotifyCollectionChanged newCol)
            newCol.CollectionChanged += ed.Zones_CollectionChanged;
        ed.RedrawZones();
    }

    private void Zones_CollectionChanged(object? sender, NotifyCollectionChangedEventArgs e) => RedrawZones();

    private void LoadSnapshot(byte[]? bytes)
    {
        if (bytes is null || bytes.Length == 0)
        {
            SnapshotImg.Source = null;
            _imgW = 1280;
            _imgH = 720;
        }
        else
        {
            try
            {
                using var ms = new MemoryStream(bytes);
                var bmp = new BitmapImage();
                bmp.BeginInit();
                bmp.CacheOption = BitmapCacheOption.OnLoad;
                bmp.StreamSource = ms;
                bmp.EndInit();
                bmp.Freeze();
                SnapshotImg.Source = bmp;
                _imgW = Math.Max(1, bmp.PixelWidth);
                _imgH = Math.Max(1, bmp.PixelHeight);
            }
            catch
            {
                SnapshotImg.Source = null;
                _imgW = 1280;
                _imgH = 720;
            }
        }
        ImageSurface.Width = _imgW;
        ImageSurface.Height = _imgH;
        RedrawZones();
        UpdateEmptyHint();
    }

    private void RedrawZones()
    {
        OverlayCanvas.Children.Clear();
        UpdateEmptyHint();
        if (Zones is null) return;

        double scale = Math.Clamp(_imgW / 1280.0, 0.55, 1.5);
        double strokeWidth = Math.Max(2.0, 2.5 * scale);
        double labelFontSize = Math.Round(18.0 * scale);
        double pillTopThreshold = 38.0 * scale;
        double pillTopOffset = 34.0 * scale;
        double pillBottomPad = 6.0 * scale;

        foreach (var z in Zones.OrderBy(item => item.IsShelf ? 1 : 0))
        {
            var (x, y, w, h) = RoiCoordinateConverter.NormalizedToPixels(
                new CameraZone { XNorm = z.XNorm, YNorm = z.YNorm, WNorm = z.WNorm, HNorm = z.HNorm },
                _imgW,
                _imgH);
            bool isShelf = z.IsShelf;
            bool isSelected = ReferenceEquals(z, SelectedZone);
            var stroke = isShelf ? Color.FromRgb(251, 146, 60) : Color.FromRgb(56, 189, 248);
            byte fillAlpha = (byte)(isSelected ? 54 : (isShelf ? 38 : 24));
            var rect = new Rectangle
            {
                Width = w,
                Height = h,
                Stroke = new SolidColorBrush(stroke),
                StrokeThickness = isSelected ? strokeWidth + 1.5 : strokeWidth,
                Fill = new SolidColorBrush(Color.FromArgb(fillAlpha, stroke.R, stroke.G, stroke.B)),
            };
            if (!isShelf)
                rect.StrokeDashArray = new DoubleCollection { 8, 4 };
            Canvas.SetLeft(rect, x);
            Canvas.SetTop(rect, y);
            OverlayCanvas.Children.Add(rect);

            string rolePrefix = isShelf ? "SHELF" : "AISLE";
            var pill = new Border
            {
                Background = new SolidColorBrush(Color.FromArgb(232, 13, 16, 21)),
                BorderBrush = new SolidColorBrush(isSelected ? Colors.White : stroke),
                BorderThickness = new Thickness(4, 1, 1, 1),
                CornerRadius = new CornerRadius(4),
                Padding = new Thickness(9, 3, 10, 3),
                Child = new TextBlock
                {
                    Text = $"[{rolePrefix}] {z.Name}",
                    Foreground = new SolidColorBrush(stroke),
                    FontSize = labelFontSize,
                    FontWeight = FontWeights.SemiBold,
                },
            };
            Canvas.SetLeft(pill, x);
            Canvas.SetTop(pill, y >= pillTopThreshold ? y - pillTopOffset : y + pillBottomPad);
            OverlayCanvas.Children.Add(pill);

            if (isSelected)
            {
                double hr = Math.Max(6.0, 7.0 * scale);
                foreach (var (hx, hy) in new[] { (x, y), (x + w, y), (x, y + h), (x + w, y + h) })
                {
                    var handle = new Ellipse
                    {
                        Width = hr * 2,
                        Height = hr * 2,
                        Fill = Brushes.White,
                        Stroke = new SolidColorBrush(stroke),
                        StrokeThickness = 2.0,
                    };
                    Canvas.SetLeft(handle, hx - hr);
                    Canvas.SetTop(handle, hy - hr);
                    OverlayCanvas.Children.Add(handle);
                }
            }
        }
    }

    private (EditableZoneViewModel? Zone, CornerHandle Corner) HitTestZones(Point pt)
    {
        if (Zones is null || Zones.Count == 0) return (null, CornerHandle.None);
        double scale = Math.Clamp(_imgW / 1280.0, 0.55, 1.5);
        double hitRadius = 16.0 * scale;

        if (SelectedZone is not null && Zones.Contains(SelectedZone))
        {
            var (sx, sy, sw, sh) = RoiCoordinateConverter.NormalizedToPixels(
                new CameraZone { XNorm = SelectedZone.XNorm, YNorm = SelectedZone.YNorm, WNorm = SelectedZone.WNorm, HNorm = SelectedZone.HNorm },
                _imgW,
                _imgH);
            if ((pt - new Point(sx, sy)).Length <= hitRadius) return (SelectedZone, CornerHandle.TopLeft);
            if ((pt - new Point(sx + sw, sy)).Length <= hitRadius) return (SelectedZone, CornerHandle.TopRight);
            if ((pt - new Point(sx, sy + sh)).Length <= hitRadius) return (SelectedZone, CornerHandle.BottomLeft);
            if ((pt - new Point(sx + sw, sy + sh)).Length <= hitRadius) return (SelectedZone, CornerHandle.BottomRight);
        }

        // Check foreground zones first (Shelf over Aisle)
        foreach (var z in Zones.OrderByDescending(item => item.IsShelf ? 1 : 0))
        {
            var (zx, zy, zw, zh) = RoiCoordinateConverter.NormalizedToPixels(
                new CameraZone { XNorm = z.XNorm, YNorm = z.YNorm, WNorm = z.WNorm, HNorm = z.HNorm },
                _imgW,
                _imgH);
            if (pt.X >= zx && pt.X <= zx + zw && pt.Y >= zy && pt.Y <= zy + zh)
                return (z, CornerHandle.None);
        }
        return (null, CornerHandle.None);
    }

    private void ClearPreviewArtifacts()
    {
        if (_previewRect is not null)
        {
            OverlayCanvas.Children.Remove(_previewRect);
            _previewRect = null;
        }
        if (_previewTag is not null)
        {
            OverlayCanvas.Children.Remove(_previewTag);
            _previewTag = null;
        }
    }

    private void Surface_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
    {
        e.Handled = true;
        ClearPreviewArtifacts();

        var raw = e.GetPosition(ImageSurface);
        var start = new Point(Math.Clamp(raw.X, 0, _imgW), Math.Clamp(raw.Y, 0, _imgH));
        _dragStart = start;
        _pushedUndoForCurrentDrag = false;

        var (hitZone, hitCorner) = HitTestZones(start);
        if (hitZone is not null)
        {
            SelectedZone = hitZone;
            var (zx, zy, zw, zh) = RoiCoordinateConverter.NormalizedToPixels(
                new CameraZone { XNorm = hitZone.XNorm, YNorm = hitZone.YNorm, WNorm = hitZone.WNorm, HNorm = hitZone.HNorm },
                _imgW,
                _imgH);
            _dragTargetZone = hitZone;
            _dragOrigRectPx = [zx, zy, zw, zh];
            _activeCorner = hitCorner;
            _dragMode = hitCorner == CornerHandle.None ? DragMode.MoveZone : DragMode.ResizeCorner;
            ImageSurface.CaptureMouse();
            return;
        }

        SelectedZone = null;
        _dragTargetZone = null;
        _dragMode = DragMode.DrawNew;

        bool isShelf = !string.Equals(ActiveDrawRole, "aisle", StringComparison.OrdinalIgnoreCase);
        var stroke = isShelf ? Color.FromRgb(251, 146, 60) : Color.FromRgb(56, 189, 248);
        var fill = isShelf ? Color.FromArgb(48, 251, 146, 60) : Color.FromArgb(42, 56, 189, 248);
        double scale = Math.Clamp(_imgW / 1280.0, 0.55, 1.5);

        _previewRect = new Rectangle
        {
            Stroke = new SolidColorBrush(stroke),
            StrokeThickness = Math.Max(2.0, 3.0 * scale),
            StrokeDashArray = isShelf ? new DoubleCollection { 6, 3 } : new DoubleCollection { 10, 4 },
            Fill = new SolidColorBrush(fill),
        };
        Canvas.SetLeft(_previewRect, start.X);
        Canvas.SetTop(_previewRect, start.Y);
        OverlayCanvas.Children.Add(_previewRect);

        _previewTag = new Border
        {
            Background = new SolidColorBrush(Color.FromArgb(235, 13, 16, 21)),
            BorderBrush = new SolidColorBrush(stroke),
            BorderThickness = new Thickness(3, 1, 1, 1),
            CornerRadius = new CornerRadius(4),
            Padding = new Thickness(8, 3, 9, 3),
            Child = new TextBlock
            {
                Text = isShelf ? "DRAWING SHELF" : "DRAWING AISLE",
                Foreground = new SolidColorBrush(stroke),
                FontSize = Math.Round(16.0 * scale),
                FontWeight = FontWeights.Bold,
            },
        };
        Canvas.SetLeft(_previewTag, start.X);
        Canvas.SetTop(_previewTag, Math.Max(0, start.Y - 30.0 * scale));
        OverlayCanvas.Children.Add(_previewTag);

        ImageSurface.CaptureMouse();
    }

    private void Surface_MouseMove(object sender, MouseEventArgs e)
    {
        if (_dragStart is not Point start) return;
        var cur = e.GetPosition(ImageSurface);
        double cx = Math.Clamp(cur.X, 0, _imgW);
        double cy = Math.Clamp(cur.Y, 0, _imgH);

        if (_dragMode == DragMode.DrawNew && _previewRect is not null)
        {
            double x = Math.Min(start.X, cx);
            double y = Math.Min(start.Y, cy);
            double x2 = Math.Max(start.X, cx);
            double y2 = Math.Max(start.Y, cy);
            Canvas.SetLeft(_previewRect, x);
            Canvas.SetTop(_previewRect, y);
            _previewRect.Width = Math.Max(1, x2 - x);
            _previewRect.Height = Math.Max(1, y2 - y);

            if (_previewTag is not null)
            {
                double scale = Math.Clamp(_imgW / 1280.0, 0.55, 1.5);
                Canvas.SetLeft(_previewTag, x);
                Canvas.SetTop(_previewTag, y >= 34.0 * scale ? y - 30.0 * scale : y + 6.0 * scale);
            }
            return;
        }

        if (_dragTargetZone is not null && _dragOrigRectPx is { Length: 4 } orig)
        {
            double dx = cx - start.X;
            double dy = cy - start.Y;
            if (!_pushedUndoForCurrentDrag && (Math.Abs(dx) >= 3 || Math.Abs(dy) >= 3))
            {
                BeforeZoneMutation?.Invoke();
                _pushedUndoForCurrentDrag = true;
            }
            if (!_pushedUndoForCurrentDrag) return;

            double ox = orig[0], oy = orig[1], ow = orig[2], oh = orig[3];
            double nx = ox, ny = oy, nw = ow, nh = oh;
            const double minPx = 16.0;

            if (_dragMode == DragMode.MoveZone)
            {
                nx = Math.Clamp(ox + dx, 0, Math.Max(0, _imgW - ow));
                ny = Math.Clamp(oy + dy, 0, Math.Max(0, _imgH - oh));
            }
            else if (_dragMode == DragMode.ResizeCorner)
            {
                double left = ox, top = oy, right = ox + ow, bottom = oy + oh;
                switch (_activeCorner)
                {
                    case CornerHandle.TopLeft:
                        left = Math.Min(cx, right - minPx);
                        top = Math.Min(cy, bottom - minPx);
                        break;
                    case CornerHandle.TopRight:
                        right = Math.Max(cx, left + minPx);
                        top = Math.Min(cy, bottom - minPx);
                        break;
                    case CornerHandle.BottomLeft:
                        left = Math.Min(cx, right - minPx);
                        bottom = Math.Max(cy, top + minPx);
                        break;
                    case CornerHandle.BottomRight:
                        right = Math.Max(cx, left + minPx);
                        bottom = Math.Max(cy, top + minPx);
                        break;
                }
                nx = Math.Clamp(left, 0, _imgW - minPx);
                ny = Math.Clamp(top, 0, _imgH - minPx);
                nw = Math.Clamp(right - nx, minPx, _imgW - nx);
                nh = Math.Clamp(bottom - ny, minPx, _imgH - ny);
            }

            var norm = RoiCoordinateConverter.PixelsToNormalized(nx, ny, nw, nh, _imgW, _imgH);
            _dragTargetZone.XNorm = norm[0];
            _dragTargetZone.YNorm = norm[1];
            _dragTargetZone.WNorm = norm[2];
            _dragTargetZone.HNorm = norm[3];
            RedrawZones();
        }
    }

    private void Surface_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
    {
        // Copy and clear _dragStart BEFORE ReleaseMouseCapture(), which synchronously fires LostMouseCapture.
        if (_dragStart is not Point start) return;
        _dragStart = null;
        var mode = _dragMode;
        bool didMutate = _pushedUndoForCurrentDrag;
        _dragMode = DragMode.None;
        _activeCorner = CornerHandle.None;
        _dragTargetZone = null;
        _dragOrigRectPx = null;
        _pushedUndoForCurrentDrag = false;

        var cur = e.GetPosition(ImageSurface);
        ClearPreviewArtifacts();
        if (ImageSurface.IsMouseCaptured)
            ImageSurface.ReleaseMouseCapture();

        if (mode == DragMode.DrawNew)
        {
            double x = Math.Clamp(Math.Min(start.X, cur.X), 0, _imgW);
            double y = Math.Clamp(Math.Min(start.Y, cur.Y), 0, _imgH);
            double w = Math.Clamp(Math.Abs(cur.X - start.X), 0, _imgW - x);
            double h = Math.Clamp(Math.Abs(cur.Y - start.Y), 0, _imgH - y);

            if (w >= 8 && h >= 8)
            {
                var norm = RoiCoordinateConverter.PixelsToNormalized(x, y, w, h, _imgW, _imgH);
                ZoneDrawn?.Invoke(norm);
            }
        }
        else if (didMutate)
        {
            ZoneModified?.Invoke();
        }
    }

    private void Surface_LostMouseCapture(object sender, MouseEventArgs e)
    {
        ClearPreviewArtifacts();
        _dragStart = null;
        _dragMode = DragMode.None;
        _dragTargetZone = null;
        _dragOrigRectPx = null;
        _pushedUndoForCurrentDrag = false;
    }
}
