using System.Collections.ObjectModel;
using System.IO;
using System.Text;
using System.Xml;
using System.Xml.Linq;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Win32;
using StoreGuard.Core.Models;
using StoreGuard.Core.Utilities;

namespace StoreGuard.Desktop.ViewModels;

public sealed record ApexNodeSummary(
    string InstanceId,
    string DisplayName,
    string ToolName,
    bool Enabled,
    int ParameterCount)
{
    public string StateBadge => Enabled ? "ON" : "OFF";
    public string Subtitle => string.IsNullOrWhiteSpace(ToolName)
        ? InstanceId
        : $"{ToolName} · {ParameterCount}p";
}

public sealed partial class ApexFlowEditorViewModel : ObservableObject
{
    private readonly Func<bool> _isOwnerUnlocked;
    private readonly Action<string, string, List<CameraZone>>? _onSyncZones;
    private readonly Func<string?> _pickOpenFile;
    private readonly Func<string, string?> _pickSaveFile;
    private readonly IReadOnlyList<string> _defaultSearchPaths;
    private readonly Func<string, string, string, bool, bool>? _confirmDialog;
    private readonly Func<string, Task<bool>>? _onSolutionSaved;
    private string _savedContent = string.Empty;
    private bool _syncingFlows;

    public event EventHandler<string>? ScrollToNodeRequested;

    public ApexFlowEditorViewModel(
        Func<bool> isOwnerUnlocked,
        Action<string, string, List<CameraZone>>? onSyncZones = null,
        Func<string?>? pickOpenFile = null,
        Func<string, string?>? pickSaveFile = null,
        IEnumerable<string>? defaultSearchPaths = null,
        Func<string, string, string, bool, bool>? confirmDialog = null,
        Func<string, Task<bool>>? onSolutionSaved = null,
        string? activeSolutionPath = null)
    {
        _isOwnerUnlocked = isOwnerUnlocked ?? throw new ArgumentNullException(nameof(isOwnerUnlocked));
        _onSyncZones = onSyncZones;
        _pickOpenFile = pickOpenFile ?? DefaultPickOpenFile;
        _pickSaveFile = pickSaveFile ?? DefaultPickSaveFile;
        _defaultSearchPaths = defaultSearchPaths?.ToList() ?? BuildDefaultSearchPaths(activeSolutionPath);
        _confirmDialog = confirmDialog;
        _onSolutionSaved = onSolutionSaved;
    }

    public ObservableCollection<ApexNodeSummary> Nodes { get; } = new();

    public ObservableCollection<string> FlowNames { get; } = new();

    [ObservableProperty]
    private string? selectedFlowName;

    [ObservableProperty]
    private string filePath = string.Empty;

    [ObservableProperty]
    private string xmlContent = string.Empty;

    [ObservableProperty]
    private bool isDirty;

    [ObservableProperty]
    private bool isValidXml = true;

    [ObservableProperty]
    private string validationMessage = "No .apex XML loaded.";

    [ObservableProperty]
    private string statusMessage = "Open or edit a .apex vision flow (Admin PIN required for edits).";

    [ObservableProperty]
    private string flowName = "No flow loaded";

    [ObservableProperty]
    private int connectionCount;

    [ObservableProperty]
    private ApexNodeSummary? selectedNode;

    [ObservableProperty]
    private bool wordWrap = true;

    partial void OnXmlContentChanged(string value)
    {
        IsDirty = !string.Equals(value, _savedContent, StringComparison.Ordinal);
        ValidateXml(value);
    }

    partial void OnSelectedFlowNameChanged(string? value)
    {
        if (_syncingFlows || string.IsNullOrWhiteSpace(XmlContent))
            return;
        ValidateXml(XmlContent);
    }

    partial void OnSelectedNodeChanged(ApexNodeSummary? value)
    {
        if (value is not null)
        {
            StatusMessage = $"Node {value.DisplayName} ({value.ToolName}, {value.ParameterCount} params, id={value.InstanceId})";
            if (!string.IsNullOrWhiteSpace(value.InstanceId))
                ScrollToNodeRequested?.Invoke(this, value.InstanceId);
        }
    }

    public bool TryLoadDefault()
    {
        if (!_isOwnerUnlocked())
            return false;
        if (!string.IsNullOrWhiteSpace(FilePath) || IsDirty)
            return true;

        foreach (var cand in _defaultSearchPaths)
        {
            if (!string.IsNullOrWhiteSpace(cand) && File.Exists(cand))
                return LoadFromPath(cand);
        }
        return false;
    }

    public bool LoadFromPath(string path)
    {
        if (!EnsureOwnerUnlocked("loading a .apex flow"))
            return false;
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
        {
            StatusMessage = $"File not found: {path}";
            return false;
        }

        var text = File.ReadAllText(path, Encoding.UTF8);
        FilePath = Path.GetFullPath(path);
        _savedContent = text;
        XmlContent = text;
        IsDirty = false;
        StatusMessage = $"Loaded {Path.GetFileName(path)} ({Nodes.Count} nodes).";
        return true;
    }

    [RelayCommand]
    private void BrowseAndLoad()
    {
        if (!EnsureOwnerUnlocked("opening a .apex flow"))
            return;
        var picked = _pickOpenFile();
        if (!string.IsNullOrWhiteSpace(picked))
            LoadFromPath(picked);
    }

    [RelayCommand]
    private void Reload()
    {
        if (!EnsureOwnerUnlocked("reloading a .apex flow"))
            return;
        if (string.IsNullOrWhiteSpace(FilePath) || !File.Exists(FilePath))
        {
            StatusMessage = "Cannot reload: no existing file path selected.";
            return;
        }
        if (IsDirty && _confirmDialog is not null)
        {
            bool discard = _confirmDialog(
                "Discard Unsaved XML Edits",
                $"Reloading '{Path.GetFileName(FilePath)}' will discard your unsaved XML edits. Continue?",
                "Reload from Disk",
                true);
            if (!discard) return;
        }
        LoadFromPath(FilePath);
    }

    [RelayCommand]
    private void FormatXml()
    {
        if (!EnsureOwnerUnlocked("formatting .apex XML"))
            return;
        if (string.IsNullOrWhiteSpace(XmlContent))
        {
            StatusMessage = "Nothing to format.";
            return;
        }

        try
        {
            var doc = XDocument.Parse(XmlContent, LoadOptions.None);
            using var sw = new Utf8StringWriter();
            using (var xw = XmlWriter.Create(sw, new XmlWriterSettings
            {
                Indent = true,
                IndentChars = "  ",
                OmitXmlDeclaration = false,
                Encoding = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false),
            }))
            {
                doc.Save(xw);
            }
            XmlContent = sw.ToString();
            StatusMessage = "Formatted XML with 2-space indentation.";
        }
        catch (XmlException ex)
        {
            IsValidXml = false;
            ValidationMessage = $"XML error at line {ex.LineNumber}, col {ex.LinePosition}: {ex.Message}";
            StatusMessage = "Format blocked: fix XML syntax error first.";
        }
    }

    [RelayCommand]
    private void Save()
    {
        if (!EnsureOwnerUnlocked("saving .apex flow"))
            return;
        if (string.IsNullOrWhiteSpace(FilePath))
        {
            SaveAs();
            return;
        }
        if (!ValidateXml(XmlContent))
        {
            StatusMessage = "Save blocked: fix XML syntax error first.";
            return;
        }

        WriteAtomicWithBackup(FilePath, XmlContent);
        _savedContent = XmlContent;
        IsDirty = false;
        StatusMessage = $"Saved {Path.GetFileName(FilePath)} (+ .bak backup).";
        if (_onSolutionSaved is not null)
            _ = NotifySolutionSavedAsync(FilePath);
    }

    [RelayCommand]
    private void SaveAs()
    {
        if (!EnsureOwnerUnlocked("saving .apex flow"))
            return;
        if (!ValidateXml(XmlContent))
        {
            StatusMessage = "Save blocked: fix XML syntax error first.";
            return;
        }

        var picked = _pickSaveFile(FilePath);
        if (string.IsNullOrWhiteSpace(picked))
            return;

        FilePath = Path.GetFullPath(picked);
        WriteAtomicWithBackup(FilePath, XmlContent);
        _savedContent = XmlContent;
        IsDirty = false;
        StatusMessage = $"Saved {Path.GetFileName(FilePath)} (+ .bak backup).";
        if (_onSolutionSaved is not null)
            _ = NotifySolutionSavedAsync(FilePath);
    }

    private async Task NotifySolutionSavedAsync(string savedPath)
    {
        if (_onSolutionSaved is null)
            return;
        try
        {
            bool ok = await _onSolutionSaved(savedPath);
            if (ok)
                StatusMessage = $"Saved {Path.GetFileName(savedPath)} and hot-reloaded live Apex Vision runtime.";
        }
        catch
        {
        }
    }

    [RelayCommand]
    private void SyncRoisToCamera()
    {
        if (!EnsureOwnerUnlocked("syncing ROIs to Camera Setup"))
            return;
        if (string.IsNullOrWhiteSpace(XmlContent) || !ValidateXml(XmlContent))
        {
            StatusMessage = "ROI sync blocked: valid .apex XML is required.";
            return;
        }

        var (streamUrl, solutionName, zones) = ApexProjectImporter.ParseApexXml(XmlContent, "cam");
        _onSyncZones?.Invoke(streamUrl, solutionName, zones);
        StatusMessage = zones.Count > 0
            ? $"Synced {zones.Count} ROI zone(s) from '{solutionName}' into Camera Setup."
            : $"Parsed '{solutionName}' successfully, but no ROI Zone Manager zones were defined.";
    }

    public bool ValidateXml(string? content)
    {
        Nodes.Clear();
        ConnectionCount = 0;

        if (string.IsNullOrWhiteSpace(content))
        {
            _syncingFlows = true;
            try
            {
                FlowNames.Clear();
                SelectedFlowName = null;
            }
            finally
            {
                _syncingFlows = false;
            }
            FlowName = "No flow loaded";
            IsValidXml = true;
            ValidationMessage = "Empty XML buffer.";
            return true;
        }

        try
        {
            var doc = XDocument.Parse(content, LoadOptions.SetLineInfo);
            var root = doc.Root;
            if (root is null)
            {
                FlowName = "Invalid XML";
                IsValidXml = false;
                ValidationMessage = "XML error: missing root element.";
                return false;
            }

            var flows = string.Equals(root.Name.LocalName, "ApexFlow", StringComparison.Ordinal)
                ? new List<XElement> { root }
                : root.Descendants("ApexFlow").ToList();

            var names = flows
                .Select(f => (string?)f.Attribute("name") ?? string.Empty)
                .Where(n => !string.IsNullOrWhiteSpace(n))
                .ToList();

            _syncingFlows = true;
            try
            {
                if (!FlowNames.SequenceEqual(names))
                {
                    FlowNames.Clear();
                    foreach (var fn in names)
                        FlowNames.Add(fn);
                }
                if (SelectedFlowName is null || !names.Contains(SelectedFlowName, StringComparer.Ordinal))
                    SelectedFlowName = names.FirstOrDefault();
            }
            finally
            {
                _syncingFlows = false;
            }

            XElement? activeFlow = null;
            if (!string.IsNullOrWhiteSpace(SelectedFlowName))
                activeFlow = flows.FirstOrDefault(f => string.Equals((string?)f.Attribute("name"), SelectedFlowName, StringComparison.Ordinal));
            activeFlow ??= flows.FirstOrDefault();

            if (flows.Count > 1 && activeFlow is not null)
                FlowName = $"{flows.Count} flows · {(string?)activeFlow.Attribute("name") ?? SelectedFlowName}";
            else
                FlowName = (string?)activeFlow?.Attribute("name")
                    ?? (string?)root.Attribute("name")
                    ?? root.Name.LocalName;

            var scope = activeFlow ?? root;
            foreach (var n in scope.Descendants("Node"))
            {
                var instId = (string?)n.Attribute("instance_id") ?? string.Empty;
                var dispName = (string?)n.Attribute("display_name")
                    ?? (string?)n.Attribute("name")
                    ?? instId;
                var toolName = (string?)n.Attribute("tool_name") ?? string.Empty;
                var enabled = !string.Equals((string?)n.Attribute("enabled"), "false", StringComparison.OrdinalIgnoreCase);
                var paramCount = n.Element("Parameters")?.Elements("Parameter").Count() ?? 0;
                Nodes.Add(new ApexNodeSummary(instId, dispName, toolName, enabled, paramCount));
            }

            ConnectionCount = scope.Descendants("Connection").Count();
            IsValidXml = true;
            ValidationMessage = flows.Count > 1
                ? $"Valid Solution ({flows.Count} flows) · {Nodes.Count} node(s) · {ConnectionCount} connection(s)"
                : $"Valid XML · {Nodes.Count} node(s) · {ConnectionCount} connection(s)";
            return true;
        }
        catch (XmlException ex)
        {
            FlowName = "Syntax error";
            IsValidXml = false;
            ValidationMessage = $"XML error at line {ex.LineNumber}, col {ex.LinePosition}: {ex.Message}";
            return false;
        }
    }

    private bool EnsureOwnerUnlocked(string action)
    {
        if (_isOwnerUnlocked())
            return true;
        StatusMessage = $"Locked: Admin PIN authentication is required before {action}.";
        return false;
    }

    private static void WriteAtomicWithBackup(string targetPath, string content)
    {
        var full = Path.GetFullPath(targetPath);
        var dir = Path.GetDirectoryName(full);
        if (!string.IsNullOrEmpty(dir))
            Directory.CreateDirectory(dir);

        if (File.Exists(full))
            File.Copy(full, full + ".bak", overwrite: true);

        var tmp = full + ".tmp";
        File.WriteAllText(tmp, content, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
        File.Move(tmp, full, overwrite: true);
    }

    private static string? DefaultPickOpenFile()
    {
        var dlg = new OpenFileDialog
        {
            Title = "Open Apex Vision Flow (.apex)",
            Filter = "Apex Flow Files (*.apex;*.xml)|*.apex;*.xml|All Files (*.*)|*.*",
        };
        return dlg.ShowDialog() == true ? dlg.FileName : null;
    }

    private static string? DefaultPickSaveFile(string currentPath)
    {
        var dlg = new SaveFileDialog
        {
            Title = "Save Apex Vision Flow As (.apex)",
            Filter = "Apex Flow Files (*.apex;*.xml)|*.apex;*.xml|All Files (*.*)|*.*",
            FileName = string.IsNullOrWhiteSpace(currentPath) ? "custom_flow.apex" : Path.GetFileName(currentPath),
        };
        return dlg.ShowDialog() == true ? dlg.FileName : null;
    }

    private static List<string> BuildDefaultSearchPaths(string? activeSolutionPath = null)
    {
        var list = new List<string>();
        if (!string.IsNullOrWhiteSpace(activeSolutionPath))
            list.Add(Path.GetFullPath(activeSolutionPath));
        var baseDir = AppContext.BaseDirectory;
        var cur = new DirectoryInfo(baseDir);
        for (int i = 0; i < 7 && cur is not null; i++, cur = cur.Parent)
        {
            list.Add(Path.Combine(cur.FullName, "runtime_data", "storeguard.apex"));
            list.Add(Path.Combine(cur.FullName, "storeguard.apex"));
            list.Add(Path.Combine(cur.FullName, "VMClone", "projects", "Theft_Recognition_RTSP.apex"));
            list.Add(Path.Combine(cur.FullName, "projects", "Theft_Recognition_RTSP.apex"));
            list.Add(Path.Combine(cur.FullName, "config", "Theft_Recognition_RTSP.apex"));
        }
        list.Add(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Downloads", "VMClone", "projects", "Theft_Recognition_RTSP.apex"));
        return list.Distinct(StringComparer.OrdinalIgnoreCase).ToList();
    }

    private sealed class Utf8StringWriter : StringWriter
    {
        public override Encoding Encoding => new UTF8Encoding(encoderShouldEmitUTF8Identifier: false);
    }
}
