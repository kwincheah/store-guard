using StoreGuard.Core.Models;
using StoreGuard.Core.Utilities;
using StoreGuard.Infrastructure.Security;
using Xunit;

namespace StoreGuard.Tests;

public sealed class CoreUtilitiesTests
{
    [Fact]
    public void RoiCoordinateConverter_RoundTripsAndClamps()
    {
        var norm = RoiCoordinateConverter.PixelsToNormalized(128, 72, 256, 360, 1280, 720);
        Assert.Equal([0.1, 0.1, 0.2, 0.5], norm);

        var zone = new CameraZone { XNorm = norm[0], YNorm = norm[1], WNorm = norm[2], HNorm = norm[3] };
        var px = RoiCoordinateConverter.NormalizedToPixels(zone, 640, 360);
        Assert.Equal((64.0, 36.0, 128.0, 180.0), px);

        var clamped = RoiCoordinateConverter.PixelsToNormalized(-50, 0, 2000, 1000, 1000, 500);
        Assert.Equal(0.0, clamped[0]);
        Assert.Equal(1.0, clamped[2]);
        Assert.Equal(1.0, clamped[3]);
    }

    [Fact]
    public void ApexProjectImporter_ParsesReferenceSolutionOrSyntheticXml()
    {
        var tempPath = Path.Combine(Path.GetTempPath(), $"sg_test_{Guid.NewGuid():N}.apex");
        File.WriteAllText(tempPath, """
<?xml version='1.0' encoding='utf-8'?>
<ApexSolution name="TestCam">
  <Flows><ApexFlow name="F"><Nodes>
    <Node tool_name="Image Source" name="01_RTSP_Camera"><Parameters>
      <Parameter name="stream_url" type="str">rtsp://admin:pw@192.168.1.50:554/unicast/c4/s1/live</Parameter>
    </Parameters></Node>
    <Node tool_name="ROI Zone Manager" name="01b_ROI_Zones"><Parameters>
      <Parameter name="zones" type="json">[{"name":"shelf_main","rect":[128,72,256,360]},{"name":"aisle_main","rect":[384,72,512,576]}]</Parameter>
    </Parameters></Node>
  </Nodes></ApexFlow></Flows>
</ApexSolution>
""");
        try
        {
            var (url, name, zones) = ApexProjectImporter.ParseApexFile(tempPath, "cam_04", 1280, 720);
            Assert.Equal("rtsp://admin:pw@192.168.1.50:554/unicast/c4/s1/live", url);
            Assert.Equal("TestCam", name);
            Assert.Equal(2, zones.Count);
            Assert.Equal("shelf", zones[0].Role);
            Assert.Equal("aisle", zones[1].Role);
            Assert.Equal(0.1, zones[0].XNorm, 3);
        }
        finally
        {
            File.Delete(tempPath);
        }
    }

    [Fact]
    public void PinHasher_And_DpapiSecretProtector_Work()
    {
        var hash = PinHasher.HashPin("5678");
        Assert.True(PinHasher.VerifyPin("5678", hash));
        Assert.False(PinHasher.VerifyPin("1234", hash));

        var protector = new DpapiSecretProtector();
        var enc = protector.Protect("rtsp://admin:pass123@10.0.0.2/live");
        Assert.StartsWith("dpapi:", enc);
        Assert.Equal("rtsp://admin:pass123@10.0.0.2/live", protector.Unprotect(enc));
    }
}
