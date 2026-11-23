using Microsoft.EntityFrameworkCore;
using StoreGuard.Core.Models;

namespace StoreGuard.Infrastructure.Persistence;

public sealed class AppDbContext : DbContext
{
    public AppDbContext(DbContextOptions<AppDbContext> options) : base(options) { }

    public DbSet<UserAccount> Users => Set<UserAccount>();
    public DbSet<CameraConfig> Cameras => Set<CameraConfig>();
    public DbSet<CameraZone> CameraZones => Set<CameraZone>();
    public DbSet<EventRecord> Events => Set<EventRecord>();
    public DbSet<EventReview> EventReviews => Set<EventReview>();
    public DbSet<AppSettingsModel> Settings => Set<AppSettingsModel>();

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        modelBuilder.Entity<UserAccount>(e =>
        {
            e.ToTable("USERS");
            e.HasKey(x => x.Id);
            e.HasIndex(x => x.Username).IsUnique();
        });

        modelBuilder.Entity<CameraConfig>(e =>
        {
            e.ToTable("CAMERAS");
            e.HasKey(x => x.Id);
            e.HasMany(x => x.Zones)
                .WithOne()
                .HasForeignKey(z => z.CameraId)
                .OnDelete(DeleteBehavior.Cascade);
        });

        modelBuilder.Entity<CameraZone>(e =>
        {
            e.ToTable("CAMERA_ZONES");
            e.HasKey(x => x.Id);
        });

        modelBuilder.Entity<EventRecord>(e =>
        {
            e.ToTable("EVENTS");
            e.HasKey(x => x.Id);
            e.HasIndex(x => x.CreatedAtUtc);
            e.HasOne(x => x.Review)
                .WithOne()
                .HasForeignKey<EventReview>(r => r.EventId)
                .OnDelete(DeleteBehavior.Cascade);
        });

        modelBuilder.Entity<EventReview>(e =>
        {
            e.ToTable("EVENT_REVIEWS");
            e.HasKey(x => x.Id);
            e.HasIndex(x => x.EventId).IsUnique();
        });

        modelBuilder.Entity<AppSettingsModel>(e =>
        {
            e.ToTable("SETTINGS");
            e.HasKey(x => x.Id);
        });
    }
}
