using System.Text;
using System.Xml.Linq;
using StoreGuard.Core.Models;
using StoreGuard.Desktop.ViewModels;
using Xunit;

namespace StoreGuard.Tests;

public sealed class ApexFlowEditorViewModelTests : IDisposable
{
    private readonly string _tempDir;

    public ApexFlowEditorViewModelTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"sg_apex_ed_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
    }

    private const string SampleApexXml = """
<?xml version="1.0" encoding="utf-8"?>
<ApexSolution version="1.0" name="SampleStoreFlow">
  <Flows>
    <ApexFlow name="MainFlow">
      <Nodes>
        <Node instance_id="node_src" display_name="01_RTSP_Source" tool_name="Image Source" enabled="true">
          <Parameters>
            <Parameter name="stream_url" type="str">rtsp://127.0.0.1/stream1</Parameter>
          </Parameters>
        </Node>
        <Node instance_id="node_roi" display_name="02_ROI_Zones" tool_name="ROI Zone Manager" enabled="true">
          <Parameters>
            <Parameter name="zones" type="json">[{"name":"shelf_1","rect":[128,72,256,180]}]</Parameter>
          </Parameters>
        </Node>
      </Nodes>
      <Connections>
        <Connection from_node="node_src" from_port="Image" to_node="node_roi" to_port="Image" />
      </Connections>
    </ApexFlow>
  </Flows>
</ApexSolution>
""";

    [Fact]
    public void LockedOwner_BlocksLoadFormatSaveAndSync()
    {
        var path = Path.Combine(_tempDir, "flow.apex");
        File.WriteAllText(path, SampleApexXml, Encoding.UTF8);

        bool unlocked = false;
        int syncCalls = 0;
        var vm = new ApexFlowEditorViewModel(
            () => unlocked,
            onSyncZones: (_, _, _) => syncCalls++);

        Assert.False(vm.TryLoadDefault());
        Assert.False(vm.LoadFromPath(path));
        Assert.Contains("Locked", vm.StatusMessage, StringComparison.OrdinalIgnoreCase);

        vm.XmlContent = "<bad>";
        vm.FormatXmlCommand.Execute(null);
        Assert.Contains("Locked", vm.StatusMessage, StringComparison.OrdinalIgnoreCase);

        vm.SaveCommand.Execute(null);
        Assert.Contains("Locked", vm.StatusMessage, StringComparison.OrdinalIgnoreCase);
        Assert.False(File.Exists(path + ".bak"));

        vm.SyncRoisToCameraCommand.Execute(null);
        Assert.Contains("Locked", vm.StatusMessage, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(0, syncCalls);
    }

    [Fact]
    public void UnlockedOwner_ValidatesSyntaxFormatsSavesBackupAndSyncsRois()
    {
        var path = Path.Combine(_tempDir, "store_flow.apex");
        File.WriteAllText(path, SampleApexXml, Encoding.UTF8);

        string? syncedUrl = null;
        string? syncedName = null;
        List<CameraZone>? syncedZones = null;

        var vm = new ApexFlowEditorViewModel(
            () => true,
            onSyncZones: (url, name, zones) =>
            {
                syncedUrl = url;
                syncedName = name;
                syncedZones = zones;
            });

        Assert.True(vm.LoadFromPath(path));
        Assert.True(vm.IsValidXml);
        Assert.False(vm.IsDirty);
        Assert.Equal("MainFlow", vm.FlowName);
        Assert.Equal(2, vm.Nodes.Count);
        Assert.Equal(1, vm.ConnectionCount);

        // Broken XML blocks save and reports line/col info
        vm.XmlContent = "<ApexSolution><Broken></ApexSolution>";
        Assert.False(vm.IsValidXml);
        Assert.True(vm.IsDirty);
        Assert.Contains("XML error", vm.ValidationMessage, StringComparison.OrdinalIgnoreCase);

        vm.SaveCommand.Execute(null);
        Assert.Contains("blocked", vm.StatusMessage, StringComparison.OrdinalIgnoreCase);
        Assert.False(File.Exists(path + ".bak"));

        // Fix XML with minified whitespace, format, save (+ .bak), and sync ROIs
        vm.XmlContent = SampleApexXml.Replace("\r\n", "").Replace("\n", "");
        Assert.True(vm.IsValidXml);

        vm.FormatXmlCommand.Execute(null);
        Assert.True(vm.IsValidXml);
        Assert.Contains("\n  <Flows>", vm.XmlContent.Replace("\r\n", "\n"));

        vm.SaveCommand.Execute(null);
        Assert.False(vm.IsDirty);
        Assert.True(File.Exists(path + ".bak"));
        Assert.Equal(2, XDocument.Load(path).Descendants("Node").Count());

        vm.SyncRoisToCameraCommand.Execute(null);
        Assert.Equal("rtsp://127.0.0.1/stream1", syncedUrl);
        Assert.Equal("SampleStoreFlow", syncedName);
        Assert.NotNull(syncedZones);
        Assert.Single(syncedZones!);
        Assert.Equal("shelf_1", syncedZones[0].Name);
        Assert.Equal("shelf", syncedZones[0].Role);
    }

    [Fact]
    public async Task MultiFlowSolution_SwitchesProceduresAndHotReloadsOnSave()
    {
        var solPath = Path.Combine(_tempDir, "storeguard.apex");
        const string multiFlowXml = """
<?xml version="1.0" encoding="utf-8"?>
<ApexSolution version="1.0" name="StoreGuard">
  <Flows>
    <ApexFlow name="cam_01">
      <Nodes>
        <Node instance_id="node_src" display_name="01_Src" tool_name="Image Source" enabled="true" />
      </Nodes>
      <Connections />
    </ApexFlow>
    <ApexFlow name="cam_02">
      <Nodes>
        <Node instance_id="node_src" display_name="01_Src" tool_name="Image Source" enabled="true" />
        <Node instance_id="node_mot" display_name="03_Motion" tool_name="Motion Gate" enabled="true" />
      </Nodes>
      <Connections>
        <Connection from_node="node_src" from_port="Image" to_node="node_mot" to_port="Image" />
      </Connections>
    </ApexFlow>
  </Flows>
</ApexSolution>
""";
        File.WriteAllText(solPath, multiFlowXml, Encoding.UTF8);

        var reloadedPaths = new List<string>();
        var vm = new ApexFlowEditorViewModel(
            () => true,
            onSolutionSaved: path =>
            {
                reloadedPaths.Add(path);
                return Task.FromResult(true);
            },
            activeSolutionPath: solPath);

        Assert.True(vm.TryLoadDefault());
        Assert.Equal(Path.GetFullPath(solPath), vm.FilePath);
        Assert.Equal(new[] { "cam_01", "cam_02" }, vm.FlowNames.ToArray());
        Assert.Equal("cam_01", vm.SelectedFlowName);
        Assert.Single(vm.Nodes);
        Assert.Contains("2 flows", vm.FlowName);

        vm.SelectedFlowName = "cam_02";
        Assert.Equal(2, vm.Nodes.Count);
        Assert.Equal(1, vm.ConnectionCount);

        vm.SaveCommand.Execute(null);
        await Task.Delay(50);
        Assert.Contains(Path.GetFullPath(solPath), reloadedPaths);
    }

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(_tempDir))
                Directory.Delete(_tempDir, recursive: true);
        }
        catch
        {
        }
    }
}
